from meta_minichess.minichess.rules import MiniChessRuleset
from scipy.stats import dlaplace
import numpy
import numpy as np

# Prob Distributions of ML samples
# Modeled after truncated normal distributino
prob_dists = [
	[.17,.20,.17,.12,.07], # pawn; m=1, SD=2
	[.17,.20,.17,.12,.07], # knight; m=1, SD=2
	[.17,.20,.17,.12,.07], # bishop; m=1, SD=2
	[.02,.07,.12,.17,.20], # rook; m=4, SD=2
	[.17,.20,.17,.12,.07], # king; m=1, SD=2
	[.17,.20,.17,.12,.07],  # queen; m=4, SD=2
]

def new_rule_set(n=1, vary_values=False, seed = None):
    '''
    Returns set of new MiniChessRuleset objects
    Will vary piece values only if option set
    '''
    np.random.seed(seed)

    rule_sets = set()
    for i in range(n):
    	move_limits = [np.random.choice(5,1,pd)[0] for pd in prob_dists]
    	sample = MiniChessRuleset.from_vector(np.array(move_limits))
    	rule_sets.add(sample)
    return rule_sets