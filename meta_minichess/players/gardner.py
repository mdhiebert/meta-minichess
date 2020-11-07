from utils import numpy_softmax
import numpy as np
from players.abstract import Player

class RandomPlayer(Player):
    def __init__(self):
        super().__init__(1225)

    def propose_action(self, board, color, action_mask):

        action_weights = np.random.rand(self.action_space_size)

        legal_actions = action_weights * action_mask
        renormalized = numpy_softmax(legal_actions)

        idx = np.argmax(renormalized)

        action = np.zeros(self.action_space_size)
        action[idx] = 1

        return action