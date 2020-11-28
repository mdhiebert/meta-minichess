from learning.alpha_zero.distributed.mcts import MCTS
import logging
from multiprocessing import Pool
import time
import numpy as np

from learning.alpha_zero.distributed.utils import run_apply_async_multiprocessing

from tqdm import tqdm

log = logging.getLogger(__name__)

MAX_MOVES = 100
class JOATArena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        pass

    def playGame(self, player1, player2, game, max_moves=MAX_MOVES):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [player2, None, player1]
        curPlayer = 1
        board = game.getInitBoard()
        it = 0
        while game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if it > MAX_MOVES:
                return 1e-4

            # lambda x: np.argmax(pmcts.getActionProb(x, temp=0))

            if players[curPlayer + 1] == 'random':
                action = game.getRandomMove(game.getCanonicalForm(board, curPlayer), curPlayer)
            elif players[curPlayer + 1] == 'greedy':
                action = game.getGreedyMove(game.getCanonicalForm(board, curPlayer), curPlayer)
            else:
                action = np.argmax(players[curPlayer + 1].getActionProb(game.getCanonicalForm(board, curPlayer)))

            valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = game.getNextState(board, curPlayer, action)

        return curPlayer * game.getGameEnded(board, curPlayer)

    def playGames(self, pnet, nnet, args, games):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num_workers = args['numWorkers']
        num = args['arenaComparePerGame']

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        fargs = []

        for game in games:
            if pnet in ['greedy', 'random']:
                fargs.extend([(pnet, MCTS(game, nnet, args), game, args['maxMoves'])] * num)
            else:
                fargs.extend([(MCTS(game, pnet, args), MCTS(game, nnet, args), game, args['maxMoves'])] * num)

        results = run_apply_async_multiprocessing(self.playGame, fargs, num_workers, desc='Arena #1')

        fargs = []

        for game in games:
            if pnet in ['greedy', 'random']:
                fargs.extend([(MCTS(game, nnet, args), pnet, game)] * num)
            else:
                fargs.extend([(MCTS(game, nnet, args), MCTS(game, pnet, args), game)] * num)

        second_results = run_apply_async_multiprocessing(self.playGame, fargs, num_workers, desc='Arena #2')

        for gameResult in results:
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
        
        for gameResult in second_results:
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws