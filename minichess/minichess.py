from minichess.state import MiniChessState
from minichess.pieces import PieceColor
from minichess.rules import MiniChessRuleset

class MiniChess:
    def __init__(self, rules: MiniChessRuleset = None, active_color = PieceColor.WHITE):
        self.state = MiniChessState(rules)
        self.active_color = active_color

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

    def immediate_states(self):
        """
            Returns all possible next states given the current state.
        """
        return self.state.possible_next_states(self.active_color)

    def display_ascii(self):
        print(str(self.state))