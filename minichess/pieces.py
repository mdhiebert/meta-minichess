from typing import List
from minichess.resources import *
import numpy as np
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

    def possible_moves(self, board) -> List:
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
        raise NotImplementedError # TODO

    def _onehot(self) -> list:
        '''
            Returns
            -------
            numpy array representing the one-hot encoding of this piece:
                [1, 0, 0, 0, 0, 0] = Pawn
                [0, 1, 0, 0, 0, 0] = Knight
                [0, 0, 1, 0, 0, 0] = Bishop
                [0, 0, 0, 1, 0, 0] = Rook
                [0, 0, 0, 0, 1, 0] = Queen
                [0, 0, 0, 0, 0, 1] = King
        '''
        raise NotImplementedError

    def vector(self) -> np.array:
        '''
            Returns
            -------
            numpy array representing the vector encoding of this piece:
                ID +
                COLOR (0 for white, 1 for black) +
                ONEHOT
        '''
        return np.array([self.id, self.color.value] + self._onehot())

    @staticmethod
    def from_vector(vector: np.array):
        '''
            Static method that intakes a vector and returns the piece corresponding
        '''
        _id = vector[0]
        color = PieceColor(vector[1])
        onehot = vector[2:]

        pieces = [
            Pawn,
            Knight,
            Bishop,
            Rook,
            Queen,
            King
        ]

        return pieces[np.argmax(onehot)](_id, color, 0, 0) # TODO this loses info, just for viz purposes

# Specific Piece Definitions

class Pawn(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, board) -> List:
        pass # TODO

    def __str__(self):
        return WHITE_PAWN if self.color == PieceColor.WHITE else BLACK_PAWN

    def _onehot(self):
        return [1, 0, 0, 0, 0, 0]

class Knight(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, board) -> List:
        pass # TODO

    def __str__(self):
        return WHITE_KNIGHT if self.color == PieceColor.WHITE else BLACK_KNIGHT

    def _onehot(self):
        return [0, 1, 0, 0, 0, 0]

class Bishop(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, board) -> List:
        pass # TODO

    def __str__(self):
        return WHITE_BISHOP if self.color == PieceColor.WHITE else BLACK_BISHOP

    def _onehot(self):
        return [0, 0, 1, 0, 0, 0]

class Rook(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, board) -> List:
        pass # TODO

    def __str__(self):
        return WHITE_ROOK if self.color == PieceColor.WHITE else BLACK_ROOK

    def _onehot(self):
        return [0, 0, 0, 1, 0, 0]

class Queen(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, board) -> List:
        pass # TODO

    def __str__(self):
        return WHITE_QUEEN if self.color == PieceColor.WHITE else BLACK_QUEEN

    def _onehot(self):
        return [0, 0, 0, 0, 1, 0]

class King(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, board) -> List:
        pass # TODO

    def __str__(self):
        return WHITE_KING if self.color == PieceColor.WHITE else BLACK_KING

    def _onehot(self):
        return [0, 0, 0, 0, 0, 1]

# Piece Actions

class MiniChessMove:
    pass