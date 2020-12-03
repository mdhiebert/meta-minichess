import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import json
import copy

from learning.alpha_zero.distributed.utils import run_apply_async_multiprocessing

import itertools

from collections import Counter

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from learning.alpha_zero.distributed.joat_arena import JOATArena as Arena
from learning.alpha_zero.distributed.mcts import MCTS

log = logging.getLogger(__name__)

class Explorer():
    """
    This class executes the self-play in order to get a sampling of states visited. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.

    Parameters
    ----------
    games :: List[Game] : the list of games to execute self play on

    probs :: List[float] : the list of probabilities with which to sample our games. probs[i] = P(games[i])

    nnet :: NNet : the network to learn with

    args :: dict : args
    """

    def __init__(self, games, probs, nnet, args):
        self.games = games
        self.probs = probs
        assert round(sum(self.probs), 6) == 1, f'Expected probabilites to sum to 1, instead summed to {sum(self.probs)}'
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.games[0], args)  # the competitor network
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
        state_counter = Counter()

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
            state_counter.update(game.stringRepresentation(board)) #count the visit to the board

            board, curPlayer = game.getNextState(board, curPlayer, action)

            r = game.getGameEnded(board, curPlayer)
            
            moves += 1

            if moves >= self.args['maxMoves']:
                r = 1e-4

            if r != 0:
                return ([(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples], state_counter)

    def explore(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It tallies the number of states visited during this iteration, and adds
        the tally to the one saved currently in the json file at args.json_path.
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args['numIters'] + 1):

            state_counts = {game.__class__.__name__: Counter() for game in self.games} 

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
                iterationTrainExamples, iter_counters = zip(*iterationTrainExamples)

                iterationTrainExamples = list(itertools.chain.from_iterable(iterationTrainExamples))
                state_counts[game.__class__.__name__] += sum(iter_counters, Counter())

                # shuffle examples before training
                shuffle(iterationTrainExamples)

                # train our network
                pi_v_losses = policy_prime.train(iterationTrainExamples)

                policies_prime.append(policy_prime.state_dict())

                for pi,v in pi_v_losses:
                    pi_sum += pi
                    v_sum += v
                    counter += 1
            
            # compute average parameters and load into self.nnet
            self.nnet.load_average_params(policies_prime)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args['checkpoint'] + '/exploring', filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'] + '/exploring', filename='temp.pth.tar')
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
                    self.nnet.load_checkpoint(folder=self.args['checkpoint'] + '/exploring', filename='temp.pth.tar')
                else:
                    log.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'] + '/exploring', filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args['checkpoint'] + '/exploring', filename='best.pth.tar')

            log.info('Iteration Complete. Writing counts to "%s/%s"...', *self.args['json_folder_file'])
            # create the json file
            path = os.path.join(self.args['json_folder_file'][0], self.args['json_folder_file'][1])
            with open(path, 'a+') as f:
                if os.stat(path).st_size == 0: ## file just created/empty
                    log.info('No counts found. Writing to empty file.')
                    old_counts = {game.__class__.__name__: Counter() for game in self.games}
                else: ## load the counts from the file
                    log.info('Loading counts...')
                    f.seek(0)
                    str_counts = f.read()
                    # print('STRING OF JSON:', type(str_counts), str_counts)
                    old_counts = json.loads(str_counts)
                    old_counts = {game: Counter(v) for game, v in old_counts.items()}
                master_counts = {game.__class__.__name__: state_counts[game.__class__.__name__]+old_counts[game.__class__.__name__] for game in self.games}
                # countiung logic: turn {gametype -> Counter} into {gametype -> {state -> count}}
                master_counts = {game: dict(counter) for game, counter in master_counts.items()}
                log.info('Writing...')
                f.truncate(0) #clear file
                json.dump(master_counts, f)
            log.info('Counts written to json file "%s/%s"...', *self.args['json_folder_file'])


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
            'BichromaticChessGame': (0, 0.5, 1),
            'Meta': (0, 0, 1)
        }

        l_dict = {
            'GardnerMiniChessGame': [],
            'BabyChessGame': [],
            'MalletChessGame': [],
            'RifleChessGame': [],
            'AtomicChessGame': [],
            'DarkChessGame': [],
            'MonochromaticChessGame': [],
            'BichromaticChessGame': [],
            'Meta': []
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

        wr,dr,lr = map(list,zip(*win_rates))

        plt.plot(wr, label='Win Rate', c='g')
        plt.plot(dr, label='Draw Rate', c='y')
        plt.plot(lr, label='Loss Rate', c='r')


        plt.title('Win/Draw/Loss Rates vs {}'.format(opponent))
        plt.xlabel('Iteration (~10^2 Games)')
        plt.ylabel('Rate')
        plt.savefig(f'results/win_rates_{opponent}.png')

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict: del self_dict['pool']
        return self_dict