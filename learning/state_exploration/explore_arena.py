from learning.alpha_zero.distributed.mcts import MCTS
import logging
from multiprocessing import Pool
import time
import numpy as np

from learning.alpha_zero.distributed.utils import run_apply_async_multiprocessing_no_visual

from tqdm import tqdm

log = logging.getLogger(__name__)

MAX_MOVES = 100
class ExploreArena():
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

        seen_states = []

        players = [player2, None, player1]
        curPlayer = 1
        board = game.getInitBoard()
        it = 0
        while game.getGameEnded(board, curPlayer) == 0:

            seen_states.append((game.stringRepresentation(board), type(game).__name__))

            it += 1
            if it > MAX_MOVES:
                return seen_states

            if players[curPlayer + 1] == 'random':
                try:
                    action = game.getRandomMove(game.getCanonicalForm(board, curPlayer), curPlayer)
                except IndexError:
                    return seen_states
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

        return seen_states

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

        oneWon = 0
        twoWon = 0
        draws = 0

        fargs = []

        for game in games:
            fargs.extend([('random', 'random', game, args['maxMoves'])] * num)

        results = run_apply_async_multiprocessing_no_visual(self.playGame, fargs, num_workers, desc='Arena #1')

        return results