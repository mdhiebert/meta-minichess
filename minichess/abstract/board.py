from minichess.abstract.piece import AbstractChessPiece, PieceColor
from minichess.abstract.action import AbstractChessAction

import numpy as np

from typing import List

class AbstractChessBoard:
    '''
        An abstract data type representing a Chess Board,
        agnostic of rules.
    '''
    def __init__(self, side_length):
        self.side_length = side_length
        self.move_history = []

    def push(self, action: AbstractChessAction):
        '''
            Push a `AbstractChessAction` to this chess board,
            applying it to the current board and updating
            current state appropriately.

            Parameters
            ----------
            action :: AbstractChessAction : an action to apply to
            this game board.
        '''
        raise NotImplementedError

    def pop(self) -> AbstractChessAction:
        '''
            Removes the most recent move from this board, undoing
            its effects and returning the move itself.

            Returns
            -------
            An AbstractChessAction for the most recent action pushed
            to this Board.
        '''
        raise NotImplementedError

    def reward(self) -> float:
        '''
            Returns
            -------
            The reward value of the current state of the board.
        '''
        raise NotImplementedError

    def legal_actions(self) -> List[AbstractChessAction]:
        '''
            Returns
            -------
            List of AbstractChessActions corresponding to all possible legal moves in current game state.
        '''

    def legal_action_mask(self) -> np.array:
        '''
            Returns
            -------
            shape (NUM_ACTIONS,) numpy array of 0s and 1s, where a 0 corresponds 
            to an illegal move, and 1 corresponds to a legal move
        '''
        raise NotImplementedError

    def state_vector(self) -> np.array:
        '''
            Outputs a vector representation of this board for use in providing
            some observation to an RL model.

            Returns
            -------
            A numpy array representing the current state of this board.
        '''
        raise NotImplementedError

    def __str__(self) -> str:
        '''
            TODO
        '''

        raise NotImplementedError

class AbstractChessTile:
    '''
        An abstract data type representing a chess tile.
    '''

    def __init__(self, color: PieceColor, position: tuple, piece: AbstractChessPiece) -> None:
        self.color = color
        self.position = position
        self.piece = piece

    def occupied(self):
        return self.piece is not None

    def push(self, piece: AbstractChessPiece):
        '''
            Push a new piece to this tile, removing old occupying pieces.

            Parameters
            ----------
            piece :: AbstractChessPiece : the piece to occupy this tile.
        '''
        self.pop() # clear former piece, if applicable

        self.piece = piece
        self.piece.set_position(self.position)

    def pop(self):
        '''
            Removes (and returns) the piece from this tile.
        '''
        
        if self.occupied():
            piece = self.piece
            self.piece = None

            piece.clear_position()
            
            return piece
        
        return None

    def peek(self):
        '''
            Returns the piece from this tile.
        '''
        return self.piece