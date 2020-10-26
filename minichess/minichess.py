from minichess.state import MiniChessState
from minichess.pieces import PieceColor
from minichess.rules import MiniChessRuleset
from learning.action import MiniChessAction
import enum
import random

TURN_CUTOFF = 100

class TerminalStatus(enum.Enum):
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW = 3

# TODO add_player and such

class MiniChess:
    def __init__(self, rules: MiniChessRuleset = None, active_color = PieceColor.WHITE):
        self.state = MiniChessState(rules)
        self.active_color = active_color

        self.turn_counter = 0

        self.terminal = False

    @staticmethod
    def init_from_rules(rules):
        '''
            Static method to initialize a minichess environment with a
            MinichessRules

            rules :: MiniChessRuleset : The rules that this environment
                exists under.
        '''
        return MiniChess(rules)

    def current_state(self):
        """
            Returns the current state of the board as a vector with onehots
            encoding each tile on the board (so shape is 25x6).

            Returns
            -------
            np array of shape (25, 6) representing the current state of the
            board
        """
        return self.state

    def immediate_states(self) -> list:
        """
            Returns all possible next states given the current state.
        """

        next_states = self.state.possible_next_states(self.active_color)

        filtered_states = filter(lambda state: not state.in_check(self.active_color), next_states)

        return list(filtered_states)

    def terminal_status(self):
        """
            Returns the status of this game.
        """

        # if the game has gone on this long, random moves could go on indefinitely
        if self.turn_counter > TURN_CUTOFF: return TerminalStatus.DRAW

        if self.state.in_check(self.active_color): # this player is in check
            if len(self.immediate_states()) == 0: # no possible moves -> checkmate!
                return TerminalStatus.WHITE_WIN if self.active_color == PieceColor.BLACK else TerminalStatus.BLACK_WIN
            else:
                return TerminalStatus.ONGOING
        else: # not in check
            if len(self.immediate_states()) == 0: # not in check but can't move
                return TerminalStatus.DRAW # stalemate
            else:
                return TerminalStatus.ONGOING # keep playing

    def apply_action(self, action: MiniChessAction):
        '''
            TODO write docstring
        '''
        self.state = self.state.apply_move(action.to_minichess_move(self.state))
        
        self.state = self.state.rotate_invert()

        self.turn_counter += 1

        self.active_color = PieceColor.invert(self.active_color)

    def play_random(self):
        """
            Method for debugging that plays out (and prints) a game with both
            players simply making random moves.
        """
        while self.terminal_status() == TerminalStatus.ONGOING:
            self.display_ascii()
            print('+-------------+')
            
            candidates = self.immediate_states()
            choice = random.randint(0, len(candidates) - 1)

            self.state = candidates[choice]

            self.turn_counter += 1

            self.active_color = PieceColor.invert(self.active_color)

        self.display_ascii()
        print('+-------------+')
        print(self.terminal_status().name)

    def display_ascii(self):
        """
            Prints the string version of the current game state.
        """
        print(str(self.state))