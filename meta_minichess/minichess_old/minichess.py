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

class MiniChess:
    def __init__(self, rules: MiniChessRuleset = None, active_color: PieceColor = PieceColor.WHITE, state: MiniChessState = None):
        self.state = MiniChessState(rules) if state == None else state
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

    @staticmethod
    def init_from_state(state: MiniChessState, active_color: PieceColor, turn_counter: int = 0):
        '''
            Initialize a game from a game state.
        '''
        
        # TODO turn_counter implementation

        return MiniChess(rules=None, active_color=active_color, state=state)

    def current_state(self):
        """
            Returns the current state of the board.

            Returns
            -------
            MiniChessState representing the current state of the board
        """
        return self.state

    def immediate_states(self) -> list:
        """
            Returns all possible next states given the current state.
        """

        next_states = self.state.possible_next_states(self.active_color)

        # filtered_states = filter(lambda state: not state.in_check(self.active_color), next_states)

        return next_states

    def terminal_status(self):
        """
            Returns the status of this game.
        """

        # if the game has gone on this long, random moves could go on indefinitely
        if self.turn_counter > TURN_CUTOFF:
            self.terminal = True
            return TerminalStatus.DRAW

        if self.state.in_check(self.active_color): # this player is in check
            if len(self.immediate_states()) == 0: # no possible moves -> checkmate!
                self.terminal = True
                return TerminalStatus.WHITE_WIN if self.active_color == PieceColor.BLACK else TerminalStatus.BLACK_WIN
            else:
                return TerminalStatus.ONGOING
        else: # not in check
            if len(self.immediate_states()) == 0: # not in check but can't move
                self.terminal = True
                return TerminalStatus.DRAW # stalemate
            else:
                return TerminalStatus.ONGOING # keep playing

    def apply_action(self, action: MiniChessAction):
        '''
            TODO write docstring
        '''
        self.state = self.state.apply_move(action.to_minichess_move(self.state))
        
        # self.state = self.state.rotate_invert() # TODO is this worth it?

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

    def make_random_move(self):
        '''
            Makes a random move for the currently active color and progresses to next turn.
        '''

        self.state = random.choice(self.immediate_states())

        self.turn_counter += 1

        self.active_color = PieceColor.invert(self.active_color)

    def make_greedy_move(self):
        '''
        '''
        raise NotImplementedError # TODO

    def rollout_to_completion(self):
        '''
            Randomly plays out this game until it reaches a terminal state.

            Returns
            -------
            TerminalStatus for the result of this game upon playing randomly.
        '''
        while self.terminal_status() == TerminalStatus.ONGOING:
            
            candidates = self.immediate_states()
            self.state = random.choice(candidates)

            self.turn_counter += 1

            self.active_color = PieceColor.invert(self.active_color)

        return self.terminal_status()

    def display_ascii(self):
        """
            Prints the string version of the current game state.
        """
        print(str(self.state))