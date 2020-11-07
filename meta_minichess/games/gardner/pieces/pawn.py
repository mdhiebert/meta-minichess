import numpy as np
from meta_minichess.games.abstract.piece import AbstractChessPiece, PieceColor
from resources import BLACK_PAWN, WHITE_PAWN

class Pawn(AbstractChessPiece):
    def __init__(self, color: PieceColor, position: tuple, value: int) -> None:
        super().__init__(color, position, value)

    def _onehot(self):
        return np.array([1, 0, 0, 0, 0, 0])

    def __str__(self):
        return WHITE_PAWN if self.color == PieceColor.WHITE else BLACK_PAWN