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

    parser = argparse.ArgumentParser(description='Test a JOAT minichess model via n-shot adaptation.')

    parser.add_argument('--loading_path', action='store', default=None, help='Path to the JOAT model weights.')

    parser.add_argument('--episodes', action='store', default='100', type=int, help='Number of episodes of self-play for adaptation og JOAT (default: 100)' )

    parser.add_argument('--mcts_sims', action='store', default='200', type=int, help='Number of MCTS simulations to perform per action.')

    parser.add_argument('--arenapergame', action='store', default='10', type=int, help='The number of Arena Games to conduct per game variant. This number will be divided in half to give the model equal reps as both black and white. If this is 0, Arena will be skipped. (default: 10)')

    parser.add_argument('--max_moves', action='store', default='75', type=int, help='The maximum number of moves permitted in a minichess game before declaring a draw (default: 75)')

    parser.add_argument('--workers', action='store', default='1', type=int, help='The number of workers to use to process self- and arena-play. A value >1 will leverage multiprocessing. (default: 1)')

    parser.add_argument('--games', dest='games', action='store', nargs='+', default='gardner', choices=['gardner', 'mallet', 'baby', 'rifle', 'dark', 'atomic'], type=str, help='The games to consider during testing. The adapted JOAT model will be assessed for each variant. (default: just gardner)')

    parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help='The learning rate during adaptation.')

    parser.add_argument('--dropout', action='store', type=float, default=0.3, help='Dropout rate during adaptation.')

    parser.add_argument('--epochs', action='store', type=int, default=10, help='Number of epochs during adaptation.')

    parser.add_argument('--batch_size', action='store', type=int, default=64, help='Batch size during adaptation.')

    parser.add_argument('--num_channels', action='store', type=int, default=512, help='Number of channels to use in the model during adaptation.')

    parser.add_argument('--eval_on_baselines', action='store_true', default=False, help='If passed in, we will evaluate our model against random and greedy players and plot the win rates.')
    
    parser.add_argument('--use_cuda', action='store_true', default=torch.cuda.is_available(), help='If passed, force the system to use CUDA. (default: whether or not CUDA is available)')

    parser.add_argument('--dont_use_cuda', action='store_true', default=False, help='Force the system NOT to use CUDA, even if its available (default: False)')

    parser.add_argument('--skip_self_play', action='store_true', default=False, help='Skip self-play to to load in training examples; if true, must be .examples path in same directory as loading_path per game in --games (default: False)')

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

    # Value Error logic
    if args.loading_path == None:
        log.info('No JOAT model found; must be passed in to assess during testing')
        raise ValueError('No JOAT model found; must be passed in via --loading_path')
    if args.arenapergame < 1:
        log.info('Must have atleast one arena iteration')
        raise ValueError('Must have atleast one arena iteration')

    # initialize args

    train_args = dict({
        'numEps': args.episodes,              # Number of complete self-play simulations to provide JOAT during adaptation.
        'tempThreshold': 15,        # ?
        'maxlenOfQueue': 200000,    # Max number of game examples provided for adaptation.
        'numMCTSSims': args.mcts_sims,  
        'arenaComparePerGame': args.arenapergame,         # Number of games to play during arena play to assess JOAT adaptation.
        'cpuct': 1,
        'maxMoves': args.max_moves, # Per game before draw.

        'numWorkers': args.workers,

        'lr': args.learning_rate, # For adaptation.
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'cuda': use_cuda,
        'num_channels': args.num_channels,

        'evalOnBaselines': args.eval_on_baselines, # If True, will compare adapted JOAT to random and greedy players as well as original JOAT.

        'checkpoint': './temp/',
        'load_model': not args.loading_path is None,
        'load_folder_file': ('/'.join(args.loading_path.split('/')[:-1]),args.loading_path.split('/')[-1]),
        'numItersForTrainExamplesHistory': 100,
        'skipSelfPlay': args.skip_self_play,
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
    in_games = [args.games] if type(args.games)==str else args.games
    games = [game_to_class[g]() for g in in_games]


    # handle imports
    if args.workers > 1:
        from learning.alpha_zero.distributed.joat_coach import JOATCoach
        from learning.alpha_zero.distributed.pytorch.NNet import NNetWrapper as nn
        from learning.alpha_zero.distributed.utils import *
        # from learning.alpha_zero.distributed.pitter import JOATPitter
    else:
        from learning.alpha_zero.undistributed.joat_coach import JOATCoach
        from learning.alpha_zero.undistributed.pytorch.NNet import NNetWrapper as nn
        from learning.alpha_zero.undistributed.utils import *
        from learning.alpha_zero.undistributed.pitter import JOATPitter

    log.info('Loading %s...', 'Minichess Variants')

    log.info('Loading %s...', nn.__name__)
    joat = nn(games[0], train_args)

    # load JOAT model

    log.info('Loading JOAT model "%s/%s"...', *train_args['load_folder_file'])
    joat.load_checkpoint(train_args['load_folder_file'][0], train_args['load_folder_file'][1])

    log.info('Loading JOAT Pitter...')

        
    p = JOATPitter(games, joat, train_args)

    if train_args['skipSelfPlay']:
        log.info("Loading 'trainExamples' per game from files...")
        for game in games:
            p.loadTrainExamples(game.__class__)

    p.adapt()

    p.joat.save_checkpoint(train_args['load_folder_file'][0], train_args['load_folder_file'][1] + '_adapted')