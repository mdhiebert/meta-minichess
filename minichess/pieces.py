from minichess.state import MiniChessState
from typing import List
import enum

class PieceColor(enum.Enum):
    WHITE = 0
    BLACK = 1

class Piece:
    '''
        The abstract data type for a mini chess piece.

        Parameters
        ----------
        _id :: int > 0 : unique identifier for this piece

        color :: PieceColor : the color of this piece

        points :: int : value of this piece

        max_move_range :: int in [0,4] : maximum number of
        steps this piece can take in any one direction.
    '''
    def __init__(self, _id: int, color: PieceColor, points = 1, max_move_range: int = 1):
        self.id = _id
        self.color = color
        self.points = points
        self.max_move_range = max_move_range

    def possible_moves(self, board: MiniChessState) -> List[MiniChessMove]:
        '''
            Given a board state, return all possible moves this piece can make.

            Parameters
            ----------
            board :: MiniChessState : 

            Returns
            -------
            list of MiniChessMove representing all possible moves this piece can
            make
        '''
        raise NotImplementedError

# Specific Piece Definitions

# Piece Actions

class MiniChessMove:
    pass