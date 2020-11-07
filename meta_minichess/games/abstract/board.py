import enum
from meta_minichess.games.abstract.piece import AbstractChessPiece, PieceColor
from resources import *

import numpy as np

from typing import List, Union, Iterable

LETTER_TO_COLUMN = {
    'e': 0,
    'd': 1,
    'c': 2,
    'b': 3,
    'a': 4
}

class AbstractBoardStatus(enum.Enum):
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW = 3

class AbstractChessBoard:
    '''
        An abstract data type representing a Chess Board,
        agnostic of rules.
    '''
    def __init__(self, side_length):
        self.side_length = side_length
        self.move_history = []
        self._board = self._init_board(side_length, side_length)
        self.active_color = PieceColor.WHITE

    def _init_board(self, height, width):
        self.height = height
        self.width = width

        board = list()

        tile_count = 0
        for row_num in range(height):
            board.append([])
            for col_num in range(width):
                # set our color
                color = PieceColor.BLACK if tile_count % 2 == 0 else PieceColor.WHITE
                
                # add a tile
                board[row_num].append(AbstractChessTile(color, (row_num, col_num), None))

                tile_count += 1

        return board


    def push(self, action):
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

    def pop(self):
        '''
            Removes the most recent move from this board, undoing
            its effects and returning the move itself.

            Returns
            -------
            An AbstractChessAction for the most recent action pushed
            to this Board.
        '''
        raise NotImplementedError

    def peek(self):
        '''
            Returns
            -------
            The most recent move from this board.
        '''
        raise NotImplementedError

    def get(self, index: Union[str, Iterable[int]]):
        '''
            Returns the tile located at `index`

            Parameters
            ----------
            index :: Iterable[Int, Int] or str : the index to retrieve. A str uses chess convention, an iterable uses python convention.

            Our board looks like this under chess convention:
            ```
                   E   D   C   B   A
                1 [ ] [ ] [ ] [ ] [ ] 1
                2 [ ] [ ] [ ] [ ] [ ] 2
                3 [ ] [ ] [ ] [ ] [ ] 3
                4 [ ] [ ] [ ] [ ] [ ] 4
                5 [ ] [ ] [ ] [ ] [ ] 5
                   E   D   C   B   A
            ```
            and like this under python convention:
            ```
                   0   1   2   3   4
                0 [ ] [ ] [ ] [ ] [ ] 0
                1 [ ] [ ] [ ] [ ] [ ] 1
                2 [ ] [ ] [ ] [ ] [ ] 2
                3 [ ] [ ] [ ] [ ] [ ] 3
                4 [ ] [ ] [ ] [ ] [ ] 4
                   0   1   2   3   4
            ```

            Returns
            -------
            The tile located at `index`.
        '''
        if type(index) == str: # chess convention
            # validate our input
            assert len(index) == 2 and str.isalpha(index[0]) and str.isdigit(index[1])

            alp,num = index[0].lower(), int(index[1])

            row = num - 1
            col = LETTER_TO_COLUMN[alp]
        else:
            # validate our input
            assert len(index) == 2

            row,col = index

        return self._board[row][col]

    def reward(self) -> float:
        '''
            Returns
            -------
            The reward value of the current state of the board.
        '''
        raise NotImplementedError

    def legal_actions(self) -> List:
        '''
            Returns
            -------
            List of AbstractChessActions corresponding to all possible legal moves in current game state.
        '''
        return self.legal_actions_for_color(self.active_color)

    def legal_actions_for_color(self, color: PieceColor) -> List:
        '''
            Parameters
            ----------
            color :: PieceColor : the color for which to check legal moves

            Returns
            -------
            List of AbstractChessActions corresponding to all possible legal moves by color `color` in current game state.
        '''
        raise NotImplementedError

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

    def is_valid_position(self, position: tuple) -> bool:
        '''
            Returns true if this position lies on the game board.
        '''

        row,col = position

        return row in range(0, self.height) and col in range(0, self.width)

    @property
    def status(self) -> AbstractBoardStatus:
        '''
            Returns the `AbstractBoardStatus` of this board.
        '''
        raise NotImplementedError

    def __str__(self) -> str:
        s = ''

        for row in self._board:
            s += SPACE.join([str(tile) for tile in row])
            s += '\n'

        return s

    def __iter__(self):
        for row_num in range(self.height):
            for col_num in range(self.width):
                yield self.get((row_num, col_num))

class AbstractChessTile:
    '''
        An abstract data type representing a chess tile.

        It has the following properties:

        color :: PieceColor : the color of the tile

        position :: tuple(int, int) : the raw formatted position of this tile, (0,0) is top left corner

        piece :: 
    '''

    def __init__(self, color: PieceColor, position: tuple, piece: AbstractChessPiece) -> None:
        self.color = color
        self.position = position
        self.piece = piece

    def occupied(self) -> bool:
        '''
            Returns
            -------
            True if there is a piece on this tile, False otherwise.
        '''
        return self.piece is not None

    def capturable(self, color: PieceColor) -> bool:
        '''
            Returns
            -------
            True if there is a piece of color != `color` on this tile, False otherwise.
        '''
        return self.piece is not None and self.piece.color != color

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

            # COMMENTED OUT BECAUSE state is important to maintain when undoing moves
            # piece.clear_position()
            
            return piece
        
        return None

    def peek(self):
        '''
            Returns the piece from this tile.
        '''
        return self.piece

    def reward(self):
        '''
            Returns the reward of this tile.
        '''
        return self.piece.reward

    def __str__(self):
        return str(self.peek()) if self.occupied() else EMPTY_TILE