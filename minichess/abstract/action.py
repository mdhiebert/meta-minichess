from minichess.abstract.piece import AbstractPiece

import numpy as np

from typing import List
from enum import Enum

class AbstractActionFlags(Enum):
    PROMOTE_ROOK = 0
    PROMOTE_KNIGHT = 1
    PROMOTE_BISHOP = 2
    PROMOTE_QUEEN = 3

    # TODO

class AbstractChessAction:
    '''
        An abstract data type representing a chess action,
        agnostic of rules.

        In Gardner MiniChess, there are 1225 possible actions. We have a 5x5 board from which to choose, and 8 possible
        directions to move any piece at most 4 tiles in that direction. We also have 9 types of underpromotion (3 types
        of move to reach last rank, 3 types of promotion for each move). This provides:
            (5 * 5)( (8 * 4) + 8 + 9) = 1225
        possible moves.
    '''
    def __init__(self, agent: AbstractPiece, from_pos: tuple, to_pos: tuple, captured_piece: AbstractPiece = None, modifier_flags: List[AbstractActionFlags] = None):
        pass # TODO

    def encode(self) -> np.array:
        '''
            Encodes this action as a onehot in a shape (1225,) numpy array.

            Returns
            -------
            A numpy array onehot representing the encoding of this action.
        '''
        raise NotImplementedError

    @staticmethod
    def decode(encoding_vector: np.array):
        raise NotImplementedError

    @staticmethod
    def parse(action_string: str):
        raise NotImplementedError

