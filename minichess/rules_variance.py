from minichess.rules import MiniChessRuleset
import scipy.stats

# if null, no seed set
RANDOM_SEED = null

# Default MLs
PAWN_MOVE_LIMIT = 1
KNIGHT_MOVE_LIMIT = 1
BISHOP_MOVE_LIMIT = 4
ROOK_MOVE_LIMIT = 4
KING_MOVE_LIMIT = 1
QUEEN_MOVE_LIMIT = 4

def new_rule_set(vary_values=False) -> MiniChessRuleset:
    '''
    Returns new MiniChessRuleset object
    Will vary piece values only if option set
    '''
    raise NotImplmentedError()
