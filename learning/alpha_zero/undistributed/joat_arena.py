import logging

from tqdm import tqdm

log = logging.getLogger(__name__)

MAX_MOVES = 100
class JOATArena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, games, display=None):
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
        self.player1 = player1
        self.player2 = player2
        self.games = games
        self.display = display

    def playGame(self, game, verbose=False):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = game.getInitBoard()
        it = 0
        while game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if it > MAX_MOVES:
                return 1e-4
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)

            if players[curPlayer + 1] == 'random':
                action = game.getRandomMove(game.getCanonicalForm(board, curPlayer), curPlayer)
            elif players[curPlayer + 1] == 'greedy':
                action = game.getGreedyMove(game.getCanonicalForm(board, curPlayer), curPlayer)
            else:
                action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))

            valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        for game in self.games:
            for _ in tqdm(range(num), desc=f"Arena.playGames {type(game).__name__} (1)"):
                gameResult = self.playGame(game, verbose=verbose)
                if gameResult == 1:
                    oneWon += 1
                elif gameResult == -1:
                    twoWon += 1
                else:
                    draws += 1

            self.player1, self.player2 = self.player2, self.player1

            for _ in tqdm(range(num), desc=f"Arena.playGames {type(game).__name__} (2)"):
                gameResult = self.playGame(game, verbose=verbose)
                if gameResult == -1:
                    oneWon += 1
                elif gameResult == 1:
                    twoWon += 1
                else:
                    draws += 1

        return oneWon, twoWon, draws

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        for game in self.games:
            for _ in tqdm(range(num), desc=f"Arena.playGames {type(game).__name__} (1)"):
                gameResult = self.playGame(game, verbose=verbose)
                if gameResult == 1:
                    oneWon += 1
                elif gameResult == -1:
                    twoWon += 1
                else:
                    draws += 1

            self.player1, self.player2 = self.player2, self.player1

            for _ in tqdm(range(num), desc=f"Arena.playGames {type(game).__name__} (2)"):
                gameResult = self.playGame(game, verbose=verbose)
                if gameResult == -1:
                    oneWon += 1
                elif gameResult == 1:
                    twoWon += 1
                else:
                    draws += 1

        return oneWon, twoWon, draws