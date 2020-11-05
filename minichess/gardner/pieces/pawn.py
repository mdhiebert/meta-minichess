import numpy as np
from minichess.abstract.piece import AbstractChessPiece, PieceColor

class Pawn(AbstractChessPiece):
    def __init__(self, color: PieceColor, position: tuple, value: int) -> None:
        super().__init__(color, position, value)

    def _onehot(self):
        return np.array([1, 0, 0, 0, 0, 0])