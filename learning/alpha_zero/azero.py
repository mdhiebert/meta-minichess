from games.dark.DarkChessGame import DarkChessGame
from learning.alpha_zero.joat_coach import JOATCoach
import logging

import coloredlogs

from learning.alpha_zero.coach import Coach
from learning.alpha_zero.pytorch.NNet import NNetWrapper as nn
from learning.alpha_zero.utils import *

from games.gardner import GardnerMiniChessGame
from games.baby import BabyChessGame
from games.mallet import MalletChessGame
from games.rifle import RifleChessGame
from games.atomic import AtomicChessGame
from games.monochromatic import MonochromaticChessGame
from games.bichromatic import BichromaticChessGame
# TODO dark

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 500,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaComparePerGame': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'maxMoves': 75,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():

    # define our game distribution
    game_probs = [
        (GardnerMiniChessGame(), 0.2),
        (BabyChessGame(), 0.2),
        (MalletChessGame(), 0.2),
        (RifleChessGame(), 0.08),
        (AtomicChessGame(), 0.08),
        (DarkChessGame(), 0.08),
        (MonochromaticChessGame(), 0.08),
        (BichromaticChessGame(), 0.08)
        # (AtomicChessGame(), 1)
    ]

    games,probs = map(list,zip(*game_probs))


    log.info('Loading %s...', 'Minichess Variants')

    log.info('Loading %s...', nn.__name__)
    nnet = nn(games[0])

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the JOAT Coach...')
    c = JOATCoach(games, probs, nnet, args)
    # c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()