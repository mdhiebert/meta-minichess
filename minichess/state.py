import numpy as np
from minichess.pieces import Piece, Pawn, Knight, Bishop, Rook, Queen, King, PieceColor, MiniChessMove
from typing import List
from minichess.rules import MiniChessRuleset

EMPTY_VECTOR = np.zeros(6)
BOARD_WIDTH  = 5
BOARD_HEIGHT = 5

ROOK_COL = 0
KNIGHT_COL = 1
BISHOP_COL = 2
QUEEN_COL = 3
KING_COL = 4

DEFAULT_RULES = MiniChessRuleset()

# GARDNER VARIANT BOARD

# PIECE IDs ARE CONSTANT:
#                        BLACK
# [15] [16] [17] [18] [19]    [♜] [♞] [♝] [♛] [♚]
# [10] [11] [12] [13] [14]    [♟︎] [♟︎] [♟︎] [♟︎] [♟︎]
# [  ] [  ] [  ] [  ] [  ] -> [  ] [  ] [  ] [ ] [  ]
# [00] [01] [02] [03] [04]    [♙] [♙] [♙] [♙] [♙]
# [05] [06] [07] [08] [09]    [♖] [♘] [♗] [♕] [♔]
#                        WHITE

# TODO add some sort of check for checkmate / stalemate / terminal condition

class MiniChessState:
    def __init__(self, rules: MiniChessRuleset = None, board = None):
        self.rules = rules
        if board == None:
            self.board = self._init_board() 
            self._populate_board(rules)
        else:
            self.board = board

    def _init_board(self) -> List[List]:
        '''
            Initializes all the tiles on the board.
        '''
        board = []
        for row in range(BOARD_HEIGHT):
            board.append(list())
            for _ in range(BOARD_WIDTH):
                board[row].append(MiniChessTile())

        return board

    def _populate_board(self, rules: MiniChessRuleset=None):
        '''
            Put the pieces on the board.
        '''
        if rules is None: self.rules = DEFAULT_RULES

        self._populate_board_for_color(rules, PieceColor.WHITE)
        self._populate_board_for_color(rules, PieceColor.BLACK)

    def _populate_board_for_color(self, rules: MiniChessRuleset, color: PieceColor):
        pawn_ml, pawn_v = self.rules.pawn_rules()
        knight_ml, knight_v = self.rules.knight_rules()
        bishop_ml, bishop_v = self.rules.bishop_rules()
        rook_ml, rook_v = self.rules.rook_rules()
        queen_ml, queen_v = self.rules.queen_rules()
        king_ml, king_v = self.rules.king_rules()
        
        # pawns
        pawn_row = BOARD_HEIGHT - 2 if color == PieceColor.WHITE else 1
        for i in range(BOARD_WIDTH):
            self.board[pawn_row][i].add_piece(
                Pawn(i if color == PieceColor.WHITE else i + 10, color, pawn_v, pawn_ml)
                )

        back_row = BOARD_HEIGHT - 1 if color == PieceColor.WHITE else 0
        
        self.board[back_row][ROOK_COL].add_piece(
            Rook(5 if color == PieceColor.WHITE else 15, color, rook_v, rook_ml)
        )

        self.board[back_row][KNIGHT_COL].add_piece(
            Knight(6 if color == PieceColor.WHITE else 16, color, knight_v, knight_ml)
        )

        self.board[back_row][BISHOP_COL].add_piece(
            Bishop(7 if color == PieceColor.WHITE else 17, color, bishop_v, bishop_ml)
        )

        self.board[back_row][QUEEN_COL].add_piece(
            Queen(8 if color == PieceColor.WHITE else 18, color, queen_v, queen_ml)
        )

        self.board[back_row][KING_COL].add_piece(
            King(9 if color == PieceColor.WHITE else 19, color, king_v, king_ml)
        )

    def copy_board(self):
        '''
        Returns
        -------
        A deep copy of the current board.
        '''
        new_board = []
        for row in range(BOARD_HEIGHT):
            new_board.append(list())
            for col in range(BOARD_WIDTH):
                new_board[row].append(self.board[row][col].copy())
        
        return new_board

    def apply_move(self, move: MiniChessMove):
        new_board = self.copy_board()

        frm,to = move.frm,move.to
        row_frm,col_frm = frm
        row_to,col_to = to

        # pick up our piece
        moving_piece = new_board[row_frm][col_frm].remove_piece()

        # capture enemy piece (if applicable)
        discard_piece = new_board[row_to][col_to].remove_piece() if move.is_capture else None

        # place our piece
        new_board[row_to][col_to].add_piece(moving_piece)

        # TODO some sort of logging / history process?

        return MiniChessState(rules = self.rules, board = new_board)

    def find_piece(self, piece: Piece) -> tuple:
        '''
            Finds a specific piece and returns its coordinates on the board,
            returns (-1,-1) if piece has been removed.

            Parameters
            ----------
            piece :: Piece: The MiniChess piece to search for

            Returns
            -------
            a tuple of ints y,x corresponding to the row and the column, respectively, 
            that this piece occupies. returns (-1, -1) if not found.
        '''
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                tile = self.board[row][col]

                if piece == tile.piece: return (row, col)
        
        return (-1, -1)

    def possible_next_states(self, for_color: PieceColor) -> list:
        '''
            Given a color, return a list of all possible next states for that color.

            Parameters
            ----------
            for_color :: PieceColor : the color able to move in this state

            Returns
            -------
            A list of MiniChessState, with each state representing a possible next state of the current state
        '''

        move_list = []

        # gather up all possible moves
        for row in self.board:
            for tile in row:
                if tile.occupied() and tile.piece.color == for_color: # there is a piece here of correct color
                    move_list.extend(tile.piece.possible_moves(self))

        next_states = [self.apply_move(move) for move in move_list]

        return next_states

    def in_check(self, for_color: PieceColor) -> bool:
        '''
            # TODO this is probably not the most efficient, but it is easy to understand -> maybe refactor if training is slow?

            Parameters
            ----------
            for_color :: PieceColor : the color of the king to check if in check

            Returns
            -------
            True if `for_color`'s king is in check, False otherwise
        '''
        next_states = self.possible_next_states(PieceColor.invert(for_color)) # get all possible next states for opponent

        filtered_states = []

        king_id = 9 if for_color == PieceColor.WHITE else 19
        dummy_king = King(king_id, for_color, 0, 0)


        # check to see if the king of this piece is in ALL next states (i.e. can't be captured next move)
        for state in next_states:
            found_king = state.find_piece(dummy_king)
            if found_king != (-1, -1): filtered_states.append(state)

        return len(next_states) != len(filtered_states)

    def __str__(self):
        s = ''

        for row in self.board:
            for tile in row:
                s += str(tile)
            s += '\n'

        return s


class MiniChessTile:
    '''
        A tile on a MiniChess board
    '''
    def __init__(self, piece=None):
        self.piece = piece

    def occupied(self) -> bool:
        '''
            Returns True if this tile is occupied, else False
        '''

        return self.piece is not None

    def vector(self) -> np.array:
        '''
            Returns the vector representation of this tile as a numpy array.

            Returns
            -------
            np array of size (6,) where vector is a onehot of the pieces in this matrix
        '''

        return self.piece.vector() if self.occupied() else EMPTY_VECTOR

    def remove_piece(self) -> Piece:
        '''
            Removes (and returns) piece from this tile.

            Returns
            -------
            Piece object of the piece that occupied this tile before removal. 
        '''
        to_return = self.piece
        self.piece = None
        return to_return

    def add_piece(self, piece: Piece):
        '''
            Adds a piece to this tile.
        '''
        if not issubclass(type(piece), Piece): raise RuntimeError('Cannot place object of type {} on a MiniChessTile'.format(type(piece)))
        self.piece = piece

    def copy(self):
        """
            Returns a copy of this piece.
        """
        return MiniChessTile(piece=None if self.piece == None else self.piece.copy())

    def __str__(self):
        piece_str = str(self.piece) if self.occupied() else ' '
        return '[{}]'.format(piece_str)