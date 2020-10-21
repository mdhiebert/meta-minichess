class MiniChess:
    def __init__(self, x):
        pass

    @staticmethod
    def init_from_rules(rules):
        '''
            Static method to initialize a minichess environment with a
            MinichessRules

            rules :: MiniChessRuleset : The rules that this environment
                exists under.
        '''
        pass

    def current_state(self):
        """
            Returns the current state of the board as a vector with onehots
            encoding each tile on the board (so shape is 25x6).

            Returns
            -------
            np array of shape (25, 6) representing the current state of the
            board
        """
        raise NotImplementedError

    def immediate_states(self):
        """
            Returns all possible next states given the current state.
        """
        raise NotImplementedError

class MiniChessState:
    def __init__(self, rules):
        pass