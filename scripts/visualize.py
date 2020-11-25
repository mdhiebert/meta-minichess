from learning.alpha_zero.distributed.mcts import MCTS
from learning.alpha_zero.distributed.pytorch.NNet import NNetWrapper
from games.gardner import GardnerMiniChessGame
from games.atomic import AtomicChessGame
from games.dark import DarkChessGame
from games.rifle import RifleChessGame
# from games.gardner.GardnerMiniChessLogic import Board
# from games.atomic.AtomicChessLogic import AtomicBoard as Board
from games.rifle.RifleChessLogic import RifleBoard as Board
import numpy as np

args = {
    'numIters': 500,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 200,          # Number of games moves for MCTS to simulate.
    'arenaComparePerGame': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'maxMoves': 75,

    'numWorkers': 8,
    'cuda': True,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
}

game = RifleChessGame()
nnet = NNetWrapper(game)
# nnet.load_checkpoint('temp', 'checkpoint_5.pth.tar')
nnet.load_checkpoint('temp', 'best.pth.tar')
player1 = MCTS(game, nnet, args)
player2 = 'random'

players = [player2, None, player1]
curPlayer = 1
board = game.getInitBoard()
it = 0

print(Board(5, board).display(1))

print()

should_flip = True

while game.getGameEnded(board, curPlayer) == 0:
    it += 1

    # lambda x: np.argmax(pmcts.getActionProb(x, temp=0))

    if players[curPlayer + 1] == 'random':
        action = game.getRandomMove(game.getCanonicalForm(board, curPlayer), curPlayer)
    elif players[curPlayer + 1] == 'greedy':
        action = game.getGreedyMove(game.getCanonicalForm(board, curPlayer), curPlayer)
    else:
        action = np.argmax(players[curPlayer + 1].getActionProb(game.getCanonicalForm(board, curPlayer)))

    valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)

    assert valids[action] != 0

    board, curPlayer = game.getNextState(board, curPlayer, action)

    if should_flip: print(Board(5, board).display(-1))
    else: print(Board(5, board).display(1))

    print()

    should_flip = not should_flip

result = curPlayer * game.getGameEnded(board, curPlayer)

print(result)