import argparse

from games.gardner import GardnerMiniChessGame
from games.baby import BabyChessGame
from games.mallet import MalletChessGame
from games.rifle import RifleChessGame
from games.dark import DarkChessGame
from games.atomic import AtomicChessGame

import torch

import coloredlogs
import logging

if __name__ == "__main__": # for multiprocessing
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Train a multitasking minichess model.')

    parser.add_argument('--loading_path', action='store', default=None, help='Path to the learning model weights.')

    parser.add_argument('--iterations', action='store', default='500', type=int, help='Number of full AlphaZero iterations to run for training (default: 500)')

    parser.add_argument('--episodes', action='store', default='100', type=int, help='Number of episodes of self-play per iteration (default: 100)' )

    parser.add_argument('--mcts_sims', action='store', default='200', type=int, help='Number of MCTS simulations to perform per action.')

    parser.add_argument('--arenapergame', action='store', default='10', type=int, help='The number of Arena Games to conduct per game variant per iteration. This number will be divided in half to give the model equal reps as both black and white. If this is 0, Arena will be skipped. (default: 10)')

    parser.add_argument('--max_moves', action='store', default='200', type=int, help='The maximum number of moves permitted in a minichess game before declaring a draw (default: 200)')

    parser.add_argument('--win_threshold', action='store', default='0.6', type=float, help='The win threshold above which a new model must reach during arena-play to become the new best model (default: 0.6)')

    parser.add_argument('--workers', action='store', default='1', type=int, help='The number of workers to use to process self- and arena-play. A value >1 will leverage multiprocessing. (default: 1)')

    parser.add_argument('--games', dest='games', action='store', nargs='+', default='gardner', choices=['gardner', 'mallet', 'baby', 'rifle', 'dark', 'atomic'], type=str, help='The games to consider during training. If more than one game is input, we will metatrain. (default: just gardner)')

    parser.add_argument('--probs', action='store', nargs='+', type=float, default=None, help='The probabilities of the games to consider during training. The ith probability corresponds to the ith game provided. If no value is provided, this defaults to a uniform distribution across the provided games. (default: uniform dist)')

    parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help='The learning rate during training.')

    parser.add_argument('--dropout', action='store', type=float, default=0.3, help='Dropout rate during training.')

    parser.add_argument('--epochs', action='store', type=int, default=10, help='Number of epochs during training.')

    parser.add_argument('--batch_size', action='store', type=int, default=64, help='Batch size during training.')

    parser.add_argument('--num_channels', action='store', type=int, default=512, help='Number of channels to use in the model during training.')

    parser.add_argument('--task_batch_size', action='store', type=int, default=4, help='The number of tasks to sample in a given metalearning iteration. Not used if len(games) <= 1. (default: 4)')

    parser.add_argument('--eval_on_baselines', action='store_true', default=False, help='If passed in, we will evaluate our model against random and greedy players and plot the win rates.')
    
    parser.add_argument('--use_cuda', action='store_true', default=torch.cuda.is_available(), help='If passed, force the system to use CUDA. (default: whether or not CUDA is available)')

    parser.add_argument('--dont_use_cuda', action='store_true', default=False, help='Force the system NOT to use CUDA, even if its available (default: False)')

    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    # logging

    log = logging.getLogger(__name__)

    if args.debug:
        coloredlogs.install(level='DEBUG')
    else:
        coloredlogs.install(level='INFO')

    # cuda logic
    use_cuda = False
    if args.use_cuda:
        use_cuda = True
    if args.dont_use_cuda:
        use_cuda = False

    if use_cuda:
        log.info('Using CUDA.')
    else:
        log.info('Not using CUDA.')

    # initialize args

    train_args = dict({
        'numIters': args.iterations,
        'numEps': args.episodes,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'updateThreshold': args.win_threshold,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'numMCTSSims': args.mcts_sims,          # Number of games moves for MCTS to simulate.
        'arenaComparePerGame': args.arenapergame,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,
        'maxMoves': args.max_moves,
        'taskBatchSize': args.task_batch_size,

        'numWorkers': args.workers,

        'lr': args.learning_rate,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'cuda': use_cuda,
        'num_channels': args.num_channels,

        'evalOnBaselines': args.eval_on_baselines,

        'checkpoint': './temp/',
        'load_model': not (args.loading_path is None),
        'load_folder_file': ('/'.join(args.loading_path.split('/')[:-1]),args.loading_path.split('/')[-1]) if not (args.loading_path is None) else '',
        'numItersForTrainExamplesHistory': 20,
    })

    # set up games

    game_to_class = {
        'gardner': GardnerMiniChessGame,
        'baby': BabyChessGame,
        'mallet': MalletChessGame,
        'rifle': RifleChessGame,
        'atomic': AtomicChessGame,
        'dark': DarkChessGame
    }

    # define our game distribution
    game_probs = []

    if args.probs == None:
        args.probs = [1.0 / float(len(args.games)) for _ in args.games]

    for game,prob in zip(args.games, args.probs):
        game_probs.append((game_to_class[game](), prob))

    games,probs = map(list,zip(*game_probs))


    # handle imports
    if args.workers > 1:
        from learning.alpha_zero.distributed.joat_coach import JOATCoach
        from learning.alpha_zero.distributed.pytorch.NNet import NNetWrapper as nn
        from learning.alpha_zero.distributed.utils import *
    else:
        from learning.alpha_zero.joat_coach import JOATCoach
        from learning.alpha_zero.pytorch.NNet import NNetWrapper as nn
        from learning.alpha_zero.utils import *

    log.info('Loading %s...', 'Minichess Variants')

    log.info('Loading %s...', nn.__name__)
    print(games, train_args)
    nnet = nn(games[0], train_args)

    if train_args['load_model']:
        log.info('Loading checkpoint "%s/%s"...', *train_args['load_folder_file'])
        nnet.load_checkpoint(train_args['load_folder_file'][0], train_args['load_folder_file'][1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the JOAT Coach...')

        
    c = JOATCoach(games, probs, nnet, train_args)
    # c = Coach(g, nnet, args)

    if train_args['load_model']:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    if len(games) > 1:
        log.info('Starting the metalearning process 🎉')
        c.metalearn()
    else:
        log.info('Starting the learning process 🎉')
        c.learn()