import unittest
from minichess.minichess import MiniChess

class TestState(unittest.TestCase):
    def setUp(self):
        self.mc = MiniChess()

    def test_invert_state(self):
        self.assertEqual(self.mc.current_state(), self.mc.current_state().rotate_invert().rotate_invert(),
                            'default state did not equal itself when rotated and inverted twice')

if __name__ == "__main__":
    unittest.main()