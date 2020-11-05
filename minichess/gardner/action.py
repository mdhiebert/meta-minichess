from typing import List
from minichess.abstract.piece import AbstractChessPiece, PieceColor
from minichess.gardner.board import GardnerChessBoard
from minichess.gardner.pieces import Pawn
from minichess.abstract.board import AbstractChessBoard
from minichess.abstract.action import AbstractActionFlags, AbstractChessAction, AbstractChessActionVisitor, visitor

class GardnerChessAction(AbstractChessAction):
    pass # TODO

class GardnerChessActionVisitor(AbstractChessActionVisitor):
    '''
        All standard chess rules, minus pawn double-move and castling
    '''
    
    @visitor(Pawn)
    def visit(self, piece: AbstractChessPiece, board: GardnerChessBoard) -> List:
        row,col = piece.position
        color = piece.color

        row_dir = 1 if color == PieceColor.BLACK else -1
        col_dir = -1 if color == PieceColor.BLACK else 1

        possible_moves = []

        # standard forward move
        forward_one_pos = (row + row_dir, col)
        possible_moves.extend(self._pawn_move_helper(piece, board, forward_one_pos))

        # forward-left capture
        forward_left_pos = (row + row_dir, col - col_dir)
        possible_moves.extend(self._pawn_move_helper(piece, board, forward_left_pos, True))

        # forward-right capture
        forward_right_pos = (row + row_dir, col + col_dir)
        possible_moves.extend(self._pawn_move_helper(piece, board, forward_right_pos, True))

        return possible_moves
    
    def _pawn_move_helper(self, piece: Pawn, board: GardnerChessBoard, new_position: tuple, is_capture = False) -> List[AbstractChessAction]:
        '''
            Helper function for pawn moves.
        '''
        possible_moves = []

        if board.is_valid_position(new_position):

            if (is_capture and board.get(new_position).capturable(piece.color)) or not (is_capture or board.get(new_position).occupied()):

                # check if this is last row
                if new_position[0] in [0, 7]: # if yes, we must promote

                    for flag in [AbstractActionFlags.PROMOTE_QUEEN, AbstractActionFlags.PROMOTE_KNIGHT,
                                    AbstractActionFlags.PROMOTE_BISHOP, AbstractActionFlags.PROMOTE_ROOK]:

                        possible_moves.append(
                            GardnerChessAction(
                                piece,
                                piece.position,
                                new_position,
                                board.get(new_position).peek() if is_capture else None,
                                [flag] + ([AbstractActionFlags.CAPTURE] if is_capture else [])
                            )
                        )

                else: # if no, just normal move
                    possible_moves.append(
                        GardnerChessAction(
                            piece,
                            piece.position,
                            new_position,
                            board.get(new_position).peek() if is_capture else None,
                            [AbstractActionFlags.CAPTURE] if is_capture else []
                        )
                    )

        return possible_moves







