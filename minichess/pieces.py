from typing import List
from minichess.resources import *
import numpy as np
import enum

class PieceColor(enum.Enum):
    WHITE = 0
    BLACK = 1

    @staticmethod
    def invert(color):
        return PieceColor.WHITE if color == PieceColor.BLACK else PieceColor.BLACK

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

    def possible_moves(self, state) -> List:
        '''
            Given a board state, return all possible moves this piece can make.

            Parameters
            ----------
            state :: MiniChessState : the current state of the board

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
                COLOR (0 for white, 1 for black) +
                ONEHOT
        '''
        return np.array([self.color.value] + self._onehot())

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

        print('Instantiating from_vector loses important information about game state, only use for visualization purposes.')

        return pieces[np.argmax(onehot)](_id, color, 0, 0)

    def _valid_move(self, state, new_pos, capturing=False):
        '''
            Helper method to determine if this pawn can in fact move
            to this new tile.
        '''
        new_row, new_col = new_pos

        board = state.board

        # outside of board
        if new_row not in range(0, 5) or new_col not in range(0, 5):
            return False

        tile = board[new_row][new_col]

        if capturing:
            return tile.occupied() and tile.piece.color != self.color # occupied and opposite color
        else:
            return not tile.occupied() # just not occupied

    def chess_notation(self):
        '''
            Returns this piece's symbol in chess notation.

            King = K, Queen = Q, Bishop = B, Knight = N, Rook = R, Pawn = ''
        '''
        return ''

    def copy(self):
        '''
            Returns a deep copy of this Piece.

            Returns
            -------
            Piece that is equal to this piece in every way.
        '''
        raise NotImplementedError

    def invert(self):
        '''
            Returns a deep copy of this Piece, but of opposite color.

            Returns
            -------
            Piece that is equal to this piece in every way except for color, which is opposite of its original value.
        '''

        return type(self)(self.id, PieceColor.invert(self.color), self.points, self.max_move_range)

    def value(self):
        '''
            Returns
            -------
            The point value of this piece.
        '''

        modifier = -2 * self.color.value + 1

        return modifier * self.points

    def __eq__(self, other):
        return type(self) == type(other) and self.color == other.color and self.id == other.id

    def __hash__(self):
        return hash((self.id, str(self)))

# Specific Piece Definitions

class Pawn(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, state) -> List:
        # TODO this does not account for varying mobility at present
        row,col = state.find_piece(self) # look for self

        if (row,col) == (-1, -1): return [] # if this piece is not on the board, return empty list

        # BLACK = 1, WHITE = 0
        # Black moves pos dir, White moves neg dir

        possible_moves = []

        back_row = self.color.value - 1
        direction = 2 * self.color.value - 1


        # move forward one
        forward_one = (row + direction, col)
        if self._valid_move(state, forward_one, capturing=False):
            possible_moves.append(
                MiniChessMove((row,col), forward_one, piece=self)
            )

            # can't move forward 2 without being able to move forward 1

            # TODO maybe get rid of this

            if row == back_row + direction: # we are a pawn in our starting row
                # move forward two
                forward_two = (row + 2*direction, col)
                if self._valid_move(state, forward_two, capturing=False):
                    possible_moves.append(
                        MiniChessMove((row,col), forward_two, piece=self)
                    )

        # top left capture
        top_left_capture = (row + direction, col + direction)
        if self._valid_move(state, top_left_capture, capturing=True):
            possible_moves.append(
                MiniChessMove((row,col), top_left_capture, piece=self, is_capture=True)
            )

        # top right capture
        top_right_capture = (row + direction, col - direction)
        if self._valid_move(state, top_right_capture, capturing=True):
            possible_moves.append(
                MiniChessMove((row,col), top_right_capture, piece=self, is_capture=True)
            )

        return possible_moves
        
    def chess_notation(self):
        return ''

    def copy(self):
        return Pawn(self.id, self.color, self.points, self.max_move_range)

    def __str__(self):
        return WHITE_PAWN if self.color == PieceColor.WHITE else BLACK_PAWN

    def _onehot(self):
        return [1, 0, 0, 0, 0, 0]

class Knight(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, state) -> List:
        # TODO this does not account for varying mobility at present
        row,col = state.find_piece(self) # look for self

        if (row,col) == (-1, -1): return [] # if this piece is not on the board, return empty list

        if self.max_move_range == 0: return [] # can't move

        possible_moves = []

        directions = [1, -1]

        for y_dir in directions:
            for x_dir in directions:
                # y major axis
                new_row, new_col = row + (y_dir * 2), col + (x_dir)

                if self._valid_move(state, (new_row, new_col)): # non capture
                    possible_moves.append(
                        MiniChessMove((row,col), (new_row,new_col), piece=self)
                    )

                if self._valid_move(state, (new_row, new_col), capturing=True):
                    possible_moves.append(
                        MiniChessMove((row,col), (new_row,new_col), piece=self, is_capture=True)
                    )

                # x major axis

                new_row, new_col = row + (y_dir), col + (2 * x_dir)

                if self._valid_move(state, (new_row, new_col)): # non capture
                    possible_moves.append(
                        MiniChessMove((row,col), (new_row,new_col), piece=self)
                    )

                if self._valid_move(state, (new_row, new_col), capturing=True):
                    possible_moves.append(
                        MiniChessMove((row,col), (new_row,new_col), piece=self, is_capture=True)
                    )

        return possible_moves
        
    def copy(self):
        return Knight(self.id, self.color, self.points, self.max_move_range)

    def __str__(self):
        return WHITE_KNIGHT if self.color == PieceColor.WHITE else BLACK_KNIGHT

    def _onehot(self):
        return [0, 1, 0, 0, 0, 0]

class Bishop(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, state) -> List:
        row,col = state.find_piece(self) # look for self

        if (row,col) == (-1, -1): return [] # if this piece is not on the board, return empty list

        if self.max_move_range == 0: return [] # can't move

        possible_moves = []

        directions = [1, -1]
        
        for x_dir in directions:
            for y_dir in directions:
                collision = False
                for magnitude in range(1, self.max_move_range + 1):
                    if not collision:
                        x_change = x_dir * magnitude
                        y_change = y_dir * magnitude

                        new_row, new_col = row + y_change, col + x_change

                        if self._valid_move(state, (new_row, new_col)): # non capture
                            possible_moves.append(
                                MiniChessMove((row,col), (new_row,new_col), piece=self)
                            )
                        else:
                            collision = True

                        if self._valid_move(state, (new_row, new_col), capturing=True):
                            possible_moves.append(
                                MiniChessMove((row,col), (new_row,new_col), piece=self, is_capture=True)
                            )

                            collision = True

        return possible_moves

    def copy(self):
        return Bishop(self.id, self.color, self.points, self.max_move_range)

    def __str__(self):
        return WHITE_BISHOP if self.color == PieceColor.WHITE else BLACK_BISHOP

    def _onehot(self):
        return [0, 0, 1, 0, 0, 0]

class Rook(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, state) -> List:
        row,col = state.find_piece(self) # look for self

        if (row,col) == (-1, -1): return [] # if this piece is not on the board, return empty list

        if self.max_move_range == 0: return [] # can't move

        possible_moves = []

        directions = [(1, 0), (-1, 0), (0, 1), (-1, 0)]

        for direction in directions:
            collision = False
            for magnitude in range(1, self.max_move_range + 1):
                if not collision:
                    x_change = direction[1] * magnitude
                    y_change = direction[0] * magnitude

                    new_row, new_col = row + y_change, col + x_change

                    if self._valid_move(state, (new_row, new_col)): # non capture
                        possible_moves.append(
                            MiniChessMove((row,col), (new_row,new_col), piece=self)
                        )
                    else:
                        collision = True

                    if self._valid_move(state, (new_row, new_col), capturing=True):
                        possible_moves.append(
                            MiniChessMove((row,col), (new_row,new_col), piece=self, is_capture=True)
                        )

                        collision = True
        
        return possible_moves

    def copy(self):
        return Rook(self.id, self.color, self.points, self.max_move_range)

    def __str__(self):
        return WHITE_ROOK if self.color == PieceColor.WHITE else BLACK_ROOK

    def _onehot(self):
        return [0, 0, 0, 1, 0, 0]

class Queen(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, state) -> List:
        # just bishop + rook

        row,col = state.find_piece(self) # look for self

        if (row,col) == (-1, -1): return [] # if this piece is not on the board, return empty list

        if self.max_move_range == 0: return [] # can't move

        possible_moves = []

        directions = [(1, 0), (-1, 0), (0, 1), (-1, 0)]

        for direction in directions:
            collision = False
            for magnitude in range(1, self.max_move_range + 1):
                if not collision:
                    x_change = direction[1] * magnitude
                    y_change = direction[0] * magnitude

                    new_row, new_col = row + y_change, col + x_change

                    if self._valid_move(state, (new_row, new_col)): # non capture
                        possible_moves.append(
                            MiniChessMove((row,col), (new_row,new_col), piece=self)
                        )
                    else:
                        collision = True

                    if self._valid_move(state, (new_row, new_col), capturing=True):
                        possible_moves.append(
                            MiniChessMove((row,col), (new_row,new_col), piece=self, is_capture=True)
                        )

                        collision = True

        directions = [1, -1]
        
        for x_dir in directions:
            for y_dir in directions:
                collision = False
                for magnitude in range(1, self.max_move_range + 1):
                    if not collision:
                        x_change = x_dir * magnitude
                        y_change = y_dir * magnitude

                        new_row, new_col = row + y_change, col + x_change

                        if self._valid_move(state, (new_row, new_col)): # non capture
                            possible_moves.append(
                                MiniChessMove((row,col), (new_row,new_col), piece=self)
                            )
                        else:
                            collision = True

                        if self._valid_move(state, (new_row, new_col), capturing=True):
                            possible_moves.append(
                                MiniChessMove((row,col), (new_row,new_col), piece=self, is_capture=True)
                            )
                            collision = True
        
        return possible_moves

    def copy(self):
        return Queen(self.id, self.color, self.points, self.max_move_range)

    def __str__(self):
        return WHITE_QUEEN if self.color == PieceColor.WHITE else BLACK_QUEEN

    def _onehot(self):
        return [0, 0, 0, 0, 1, 0]

class King(Piece):
    def __init__(self, _id, color, points, max_move_range):
        super().__init__(_id, color, points, max_move_range)

    def possible_moves(self, state) -> List:
        row,col = state.find_piece(self) # look for self

        if (row,col) == (-1, -1): return [] # if this piece is not on the board, return empty list

        if self.max_move_range == 0: return [] # can't move

        possible_moves = []
        
        for y_diff in [-1, 0, 1]:
            for x_diff in [-1, 0, 1]:
                if x_diff == 0 and y_diff == 0: continue

                new_row, new_col = row + y_diff, col + x_diff

                if self._valid_move(state, (new_row, new_col)): # non capture
                    possible_moves.append(
                        MiniChessMove((row,col), (new_row,new_col), piece=self)
                    )

                if self._valid_move(state, (new_row, new_col), capturing=True):
                    possible_moves.append(
                        MiniChessMove((row,col), (new_row,new_col), piece=self, is_capture=True)
                    )

        return possible_moves 

    def copy(self):
        return King(self.id, self.color, self.points, self.max_move_range)

    def __str__(self):
        return WHITE_KING if self.color == PieceColor.WHITE else BLACK_KING

    def _onehot(self):
        return [0, 0, 0, 0, 0, 1]

# Piece Actions

class MiniChessMove:
    '''
        ADT representing a move.

        Parameters
        ----------
        frm :: tuple(int, int) : the coordinates of the piece's orginal position

        to :: tuple(int, int) : the coordinates of the piece's new position

        piece :: Piece : the piece that is being moved

        is_capture :: bool : whether or not this move is a capture

        is_castle :: bool : whether or not this move is a castle

        is_check :: bool : whether or not this move puts the opponent king in check
    '''
    
    def __init__(self, frm:tuple, to:tuple, piece:Piece = None, is_capture:bool = False, is_castle:bool = False, is_check:bool = False):
        self.frm = frm
        self.to = to

        self.piece = piece

        self.is_castle = is_castle
        self.is_capture = is_capture
        self.is_check = is_check

    def __str__(self):
        '''
            Returns a string representing this move in traditional chess notation.
        '''

        if self.is_castle:
            raise NotImplementedError

        raise NotImplementedError

    def __eq__(self, other):
        return type(other) == type(self) and self.frm == other.frm and self.to == other.to

