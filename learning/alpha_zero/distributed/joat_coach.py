import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

from learning.alpha_zero.distributed.utils import run_apply_async_multiprocessing

import itertools

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from learning.alpha_zero.distributed.joat_arena import JOATArena as Arena
from learning.alpha_zero.distributed.mcts import MCTS

from multiprocessing import Pool

import copy

log = logging.getLogger(__name__)

class JOATCoach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.

    Parameters
    ----------
    games :: List[Game] : the list of games to train on

    probs :: List[float] : the list of probabilities with which to sample our games. probs[i] = P(games[i])

    nnet :: NNet : the network to learn with

    args :: dict : args
    """

    def __init__(self, games, probs, nnet, args):
        self.games = games
        self.probs = probs
        assert round(sum(self.probs), 6) == 1, f'Expected probabilites to sum to 1, instead summed to {sum(self.probs)}'
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.games[0])  # the competitor network
        self.args = args
        self.mcts = None
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self, mcts, game, args):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        moves = 0

        while True:
            episodeStep += 1
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < self.args['tempThreshold'])

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = game.getNextState(board, curPlayer, action)

            r = game.getGameEnded(board, curPlayer)
            
            moves += 1

            if moves >= self.args['maxMoves']:
                r = 1e-4

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        losses = []
        rwinrates = []
        gwinrates = []

        for i in range(1, self.args['numIters'] + 1):


            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            
            game = np.random.choice(self.games, p=self.probs)

            log.info(f'Sampled game {type(game).__name__} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args['maxlenOfQueue'])

                iterationTrainExamples = run_apply_async_multiprocessing(self.executeEpisode, [(MCTS(game, self.nnet, self.args), type(game)(), self.args.copy())] * self.args['numEps'], self.args['numWorkers'], desc='Self Play')

                iterationTrainExamples = list(itertools.chain.from_iterable(iterationTrainExamples))

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args['numItersForTrainExamplesHistory']:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            pmcts = MCTS(game, self.pnet, self.args)

            pi_v_losses = self.nnet.train(trainExamples)

            for pi,v in pi_v_losses:
                losses.append((pi, v, type(game).__name__))

            self.plot_current_progress(losses)

            if self.args['arenaComparePerGame'] > 0:
                # ARENA
                nmcts = MCTS(game, self.nnet, self.args)

                log.info('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena()
                pwins, nwins, draws = arena.playGames(self.pnet, self.nnet, self.args, self.games)

                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
                if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args['updateThreshold']:
                    log.info('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
                else:
                    log.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='best.pth.tar')

            if self.args['evalOnBaselines']:
                mod_args = self.args.copy()
                mod_args['arenaComparePerGame'] = 20

                log.info('Evaluating against baselines...')

                arena = Arena()
                pwins, nwins, draws = arena.playGames('random', self.nnet, mod_args, self.games)
                rwinrates.append(float(nwins) / float(pwins + nwins + draws))
                self.plot_win_rate(rwinrates, 'Random')

                arena = Arena()
                pwins, nwins, draws = arena.playGames('random', self.nnet, mod_args, self.games)
                gwinrates.append(float(nwins) / float(pwins + nwins + draws))
                self.plot_win_rate(gwinrates, 'Greedy')

    def metalearn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        avg_losses = []
        rwinrates = []
        gwinrates = []

        for i in range(1, self.args['numIters'] + 1):

            policies_prime = []
            pi_sum = 0
            v_sum = 0
            counter = 0
            
            # bookkeeping
            log.info(f'Starting Meta-Iteration #{i} ...')

            # for task in tasks...
            for _ in range(self.args['taskBatchSize']):

                # create deepcopy for training a theta'
                policy_prime = copy.deepcopy(self.nnet)
            
                # sample a game (task)
                game = np.random.choice(self.games, p=self.probs)
                log.info(f'Sampled game {type(game).__name__} ...')

                # multiprocess to get our training examples
                iterationTrainExamples = deque([], maxlen=self.args['maxlenOfQueue'])
                iterationTrainExamples = run_apply_async_multiprocessing(self.executeEpisode, [(MCTS(game, self.nnet, self.args), type(game)(), self.args.copy())] * self.args['numEps'], self.args['numWorkers'], desc='Self Play')
                iterationTrainExamples = list(itertools.chain.from_iterable(iterationTrainExamples))

                # shuffle examples before training
                shuffle(iterationTrainExamples)

                # train our network
                pi_v_losses = policy_prime.train(iterationTrainExamples)

                policies_prime.append(policies_prime.state_dict())

                for pi,v in pi_v_losses:
                    pi_sum += pi
                    v_sum += v
                    counter += 1
            
            # compute average parameters and load into self.nnet
            self.nnet.load_average_params(policies_prime)

            # compute average losses
            avg_losses.append((float(pi_sum) / counter, float(v_sum) / counter, ''))
            self.plot_current_progress(avg_losses)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args['checkpoint'] + '/meta', filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'] + '/meta', filename='temp.pth.tar')
            pmcts = MCTS(self.games[0], self.pnet, self.args)


            # Arena if we choose to run it
            if self.args['arenaComparePerGame'] > 0:
                # ARENA
                nmcts = MCTS(self.games[0], self.nnet, self.args)

                log.info('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena()
                pwins, nwins, draws = arena.playGames(self.pnet, self.nnet, self.args, self.games)

                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
                if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args['updateThreshold']:
                    log.info('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
                else:
                    log.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='best.pth.tar')

            # our baselines if we choose to run them
            if self.args['evalOnBaselines']:
                mod_args = self.args.copy()
                mod_args['arenaComparePerGame'] = 20

                log.info('Evaluating against baselines...')

                arena = Arena()
                pwins, nwins, draws = arena.playGames('random', self.nnet, mod_args, self.games)
                rwinrates.append(float(nwins) / float(pwins + nwins + draws))
                self.plot_win_rate(rwinrates, 'Random')

                arena = Arena()
                pwins, nwins, draws = arena.playGames('greedy', self.nnet, mod_args, self.games)
                gwinrates.append(float(nwins) / float(pwins + nwins + draws))
                self.plot_win_rate(gwinrates, 'Greedy')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def plot_current_progress(self, losses):
        def uniqueish_color(n):
            """https://stackoverflow.com/a/17241345"""
            return plt.cm.gist_ncar(np.random.random(n))

        c_dict = {
            'GardnerMiniChessGame': (1, 0, 0),
            'BabyChessGame': (1, 0.5, 0),
            'MalletChessGame': (1, 1, 0),
            'RifleChessGame': (0.5, 1, 0),
            'AtomicChessGame': (0, 1, 0),
            'DarkChessGame': (0, 1, 0.5),
            'MonochromaticChessGame': (0, 1, 1),
            'BichromaticChessGame': (0, 0.5, 1)
        }

        l_dict = {
            'GardnerMiniChessGame': [],
            'BabyChessGame': [],
            'MalletChessGame': [],
            'RifleChessGame': [],
            'AtomicChessGame': [],
            'DarkChessGame': [],
            'MonochromaticChessGame': [],
            'BichromaticChessGame': []
        }

        plt.cla()
        plt.clf()

        for x,loss in enumerate(losses):
            pi,v,name = loss

            l_dict[name].append((x,pi,v,c_dict[name]))

        for name,game in l_dict.items():
            if len(game) == 0: continue
            xs,ys,_,cs = map(list, zip(*game))
            plt.scatter(xs,ys,c=cs, label=name, s=10)

        plt.xlabel('Epochs (~10^2 Games)')
        plt.ylabel('Policy Loss')
        plt.legend()
        plt.savefig('results/policy_loss.png')

        plt.cla()
        plt.clf()

        for name,game in l_dict.items():
            if len(game) == 0: continue
            xs,_,ys,cs = map(list, zip(*game))
            plt.scatter(xs,ys,c=cs, label=name, s=10)

        plt.xlabel('Epochs (~10^2 Games)')
        plt.ylabel('Value Loss')
        plt.legend()
        plt.savefig('results/value_loss.png')

    def plot_win_rate(self, win_rates, opponent):

        plt.cla()
        plt.clf()

        print(win_rates)

        plt.plot(win_rates, c='r')

        plt.title('Win Rate vs {}'.format(opponent))
        plt.xlabel('Iteration (~10^2 Games)')
        plt.ylabel('Win Rate')
        plt.savefig(f'results/win_rates_{opponent}.png')

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict: del self_dict['pool']
        return self_dict