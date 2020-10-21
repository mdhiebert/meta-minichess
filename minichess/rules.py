import numpy as np

# DEFAULTS
PAWN_MOVE_LIMIT = 1
KNIGHT_MOVE_LIMIT = 1
BISHOP_MOVE_LIMIT = 4
ROOK_MOVE_LIMIT = 4
KING_MOVE_LIMIT = 1
QUEEN_MOVE_LIMIT = 4

# From Bhuvaneswaran paper:
PAWN_VALUE = 100
KNIGHT_VALUE = 280
BISHOP_VALUE = 320
ROOK_VALUE = 479
QUEEN_VALUE = 929
KING_VALUE = 6000

class MiniChessRuleset:
    '''
        MiniChessRuleset object representing the rules of this Chess Game.
    '''
    def __init__(self, pawn_move_limit=PAWN_MOVE_LIMIT, pawn_value=PAWN_VALUE,
                        knight_move_limit=KNIGHT_MOVE_LIMIT, knight_value=KNIGHT_VALUE,
                        bishop_move_limit=BISHOP_MOVE_LIMIT, bishop_value=BISHOP_VALUE,
                        rook_move_limit=ROOK_MOVE_LIMIT, rook_value=ROOK_VALUE,
                        king_move_limit=KING_MOVE_LIMIT, king_value=KING_VALUE,
                        queen_move_limit=QUEEN_MOVE_LIMIT, queen_value=QUEEN_VALUE):
        self.pawn_move_limit = pawn_move_limit
        self.knight_move_limit = knight_move_limit
        self.bishop_move_limit = bishop_move_limit
        self.rook_move_limit = rook_move_limit
        self.king_move_limit = king_move_limit
        self.queen_move_limit = queen_move_limit

        self.pawn_value = pawn_value
        self.knight_value = knight_value
        self.bishop_value = bishop_value
        self.rook_value = rook_value
        self.king_value = king_value
        self.queen_value = queen_value

    # @staticmethod # TODO this should be static? or maybe not
    def from_vector(self, vector: np.array):
        '''
            Create a MiniChessRuleset from a vector of form:

            [PAWN_ML, KNIGHT_ML, BISHOP_ML, ROOK_ML, KING_ML, QUEEN_ML] (6,)

            or

            [
                [PAWN_ML, KNIGHT_ML, BISHOP_ML, ROOK_ML, KING_ML, QUEEN_ML],
                [PAWN_V, KNIGHT_V, BISHOP_V, ROOK_V, KING_V, QUEEN_V]
            ] (2,6)

            where ML = move limit and V = value

            Returns
            -------
            None
        '''
        f_cycle = [
            self._set_pawn_rules,
            self._set_knight_rules,
            self._set_bishop_rules,
            self._set_rook_rules,
            self._set_king_rules,
            self._set_queen_rules
        ]

        v_cycle = [
            PAWN_MOVE_LIMIT,
            KNIGHT_MOVE_LIMIT,
            BISHOP_MOVE_LIMIT,
            ROOK_MOVE_LIMIT,
            KING_MOVE_LIMIT,
            QUEEN_MOVE_LIMIT
        ]

        if vector.shape == (6,): # just move limits
            for f,ml,v in zip(f_cycle, vector, v_cycle):
                f(ml,v) # update rule values
        elif vector.shape == (2,6): # move limits and values
            for f,rule in zip(f_cycle, vector.T):
                ml,v = rule
                f(ml,v) # update rule values
        else:
            raise RuntimeError('from_vector expected input vector of shape (6,) or (2,6) but got vector of shape {}'.format(vector.shape))

    def pawn_rules(self):
        return (self.pawn_move_limit, self.pawn_value)

    def _set_pawn_rules(self, ml, v):
        self.pawn_move_limit = ml
        self.pawn_value = v

    def knight_rules(self):
        return (self.knight_move_limit, self.knight_value)

    def _set_knight_rules(self, ml, v):
        self.knight_move_limit = ml
        self.knight_value = v

    def bishop_rules(self):
        return (self.bishop_move_limit, self.bishop_value)
    
    def _set_bishop_rules(self, ml, v):
        self.bishop_move_limit = ml
        self.bishop_value = v

    def rook_rules(self):
        return (self.rook_move_limit, self.rook_value)

    def _set_rook_rules(self, ml, v):
        self.rook_move_limit = ml
        self.rook_value = v

    def king_rules(self):
        return (self.king_move_limit, self.king_value)

    def _set_king_rules(self, ml, v):
        self.king_move_limit = ml
        self.king_value = v

    def queen_rules(self):
        return (self.queen_move_limit, self.queen_value)

    def _set_queen_rules(self, ml, v):
        self.queen_move_limit = ml
        self.queen_value = v