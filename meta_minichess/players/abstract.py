import numpy as np

class Player:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size

    def propose_action(self, board, color, action_mask):
        '''
            Make an action given the current game board.

            Paramters
            ---------
            board :: AbstractChessBoard : the current game board

            color :: PieceColor : the color of the player to make a move from the perspective of

            action_mask :: np.array : the mask of legal actions
        '''
        raise NotImplementedError