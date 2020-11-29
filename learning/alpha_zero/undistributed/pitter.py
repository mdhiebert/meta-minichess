import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from learning.alpha_zero.undistributed.joat_arena import JOATArena as Arena
from learning.alpha_zero.undistributed.mcts import MCTS

log = logging.getLogger(__name__)

class JOATPitter():
    """
    This class executes the self-play + JOAT adaptation + JOAT evaluation. It uses the functions defined
    in Game and MetaMCNet. args are specified in main.py.

    Parameters
    ----------
    games :: List[Game] : the list of games to evaluate with

    joat :: NNet : the JOAT network

    args :: dict : args
    """

    def __init__(self, games, joat, args):
        self.games = games
        self.joat = joat
        self.adapt_joat = joat
        self.args = args
        self.mcts = None
        # self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def executeEpisode(self, game):
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
        self.curPlayer = 1
        episodeStep = 0

        moves = 0

        while True:
            episodeStep += 1
            canonicalBoard = game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = game.getNextState(board, self.curPlayer, action)

            r = game.getGameEnded(board, self.curPlayer)
            
            moves += 1

            if moves >= self.args.maxMoves:
                r = 1e-4

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def adapt(self):
        """
        Performs numEps episodes of self-play, and retrains the JOAT NN with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the original JOAT, as well
        as random & greedy baseline players (if eval_on_baselines==True).
        """

        for game in self.games:
            losses = []
            joatwinrates = []
            rwinrates = []
            gwinrates = []

            # bookkeeping
            log.info(f'Self-playing game {type(game).__name__} ...')

            # run self play on game variante
            variationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(game, self.joat, self.args)  # reset search tree
                variationTrainExamples += self.executeEpisode(game)

            # save the iteration examples to the history 
            self.trainExamplesHistory.append(variationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(type(game).__name__)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.joat.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            joatmcts = MCTS(game, self.joat, self.args)

            pi_v_losses = self.joat.train(trainExamples)

            for pi,v in pi_v_losses:
                losses.append((pi, v, type(game).__name__))

            self.plot_current_progress(losses)

            adapt_joatmcts = MCTS(game, self.adapt_joat, self.args)

            # ARENA

            log.info('PITTING ADAPTED AGAINST ORIGINAL JOAT')
            arena = Arena(lambda x: np.argmax(joatmcts.getActionProb(x, temp=0)),
                        lambda x: np.argmax(adapt_joatmcts.getActionProb(x, temp=0)), [game])
            pwins, nwins, draws = arena.playGames(self.args.arenaComparePerGame)
            joatwinrates.append(float(nwins) / float(pwins + nwins + draws))
            self.plot_win_rate(joatwinrates, 'Original JOAT')

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))

            if self.args.evalOnBaselines:
                arena = Arena('random',
                            lambda x: np.argmax(adapt_joatmcts.getActionProb(x, temp=0)), [game])
                pwins, nwins, draws = arena.playGames(self.args.arenaComparePerGame)
                rwinrates.append(float(nwins) / float(pwins + nwins + draws))
                self.plot_win_rate(rwinrates, 'Random')

                arena = Arena('greedy',
                            lambda x: np.argmax(adapt_joatmcts.getActionProb(x, temp=0)), [game])
                pwins, nwins, draws = arena.playGames(self.args.arenaComparePerGame)
                gwinrates.append(float(nwins) / float(pwins + nwins + draws))
                self.plot_win_rate(gwinrates, 'Greedy')


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
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

        plt.plot(win_rates, c='r')

        plt.title('Win Rate vs {}'.format(opponent))
        plt.xlabel('Iteration (~10^2 Games)')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.savefig(f'results/win_rates_{opponent}.png')