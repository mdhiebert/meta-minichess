import numpy as np
from minichess.pieces import Piece, Pawn, Knight, Bishop, Rook, Queen, King, PieceColor
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

class MiniChessState:
    def __init__(self, rules: MiniChessRuleset = None):
        self.board = self._init_board()
        self._populate_board(rules)

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
        self._populate_board_for_color(rules, PieceColor.WHITE)
        self._populate_board_for_color(rules, PieceColor.BLACK)

    def _populate_board_for_color(self, rules: MiniChessRuleset, color: PieceColor):
        if rules is None: rules = DEFAULT_RULES

        pawn_ml, pawn_v = rules.pawn_rules()
        knight_ml, knight_v = rules.knight_rules()
        bishop_ml, bishop_v = rules.bishop_rules()
        rook_ml, rook_v = rules.rook_rules()
        queen_ml, queen_v = rules.queen_rules()
        king_ml, king_v = rules.king_rules()
        
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


    def find_piece(self, piece):
        pass # TODO

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

        self.piece = piece

    def __str__(self):
        piece_str = str(self.piece) if self.occupied() else ' '
        return '[{}]'.format(piece_str)