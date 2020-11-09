from meta_minichess.games.gardner.pieces import rook
from meta_minichess.games.abstract.piece import PieceColor
from meta_minichess.games.gardner.pieces import Pawn, King, Rook
import unittest
from meta_minichess.games.gardner.board import GardnerChessBoard

GENERIC_PAWN = Pawn(PieceColor.WHITE, (-1, -1), 100)
GENERIC_KING = King(PieceColor.WHITE, (-1, -1), 6000)
GENERIC_KING_BLACK = King(PieceColor.BLACK, (-1, -1), 6000)
GENERIC_ROOK_BLACK = Rook(PieceColor.BLACK, (-1, -1), 550)

class TestGardner(unittest.TestCase):
    def setUp(self):
        self.g = GardnerChessBoard()

    def test_wipe_board(self):
        self.g.wipe_board()

        assert self.g.is_empty() == True, 'Expected board to be empty after wiping, but was not.'

        self.g.get((2,2)).push(GENERIC_PAWN)

        assert self.g.is_empty() == False, 'Expected board to not be empty after placing piece, but it was.'

    def test_king_actions_alone(self):
        self.g.wipe_board()

        self.g.get((2, 2)).push(GENERIC_KING)

        print('Testing on board...')
        print(self.g)
        print('')

        actions = self.g.legal_actions_for_color(PieceColor.WHITE)

        actions = set(actions)

        assert len(actions) == 8, 'Expected possible king actions on empty board to be set of length 8 but had length {}. Actions are:\n{}'.format(len(actions), '\n'.join([str(s) for s in actions]))

    def test_king_actions_could_have_check(self):
        
        self.g.wipe_board()

        self.g.get((0, 3)).push(GENERIC_KING)

        self.g.get((2, 4)).push(GENERIC_ROOK_BLACK)
        self.g.get((3, 3)).push(GENERIC_KING_BLACK)

        print('Testing on board...')
        print(self.g)
        print('')

        actions = self.g.legal_actions_for_color(PieceColor.BLACK)

        actions = set(actions)
        
        assert len(actions) == 15, 'Expected possible actions for black to be set of length 15 but had length {}. Actions are:\n{}'.format(len(actions), '\n'.join([str(s) for s in actions]))

        actions = self.g.legal_actions_for_color(PieceColor.WHITE)

        actions = set(actions)
        
        assert len(actions) == 3, 'Expected possible actions for white to be set of length 3 but had length {} Actions are:\n{}'.format(len(actions), '\n'.join([str(s) for s in actions]))

if __name__ == "__main__":
    unittest.main()