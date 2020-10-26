from minichess.minichess import MiniChess
from minichess.pieces import PieceColor

# TODO check / checkmate / terminal states
# TODO point values
# TODO generating action options

if __name__ == "__main__":
    mc = MiniChess.init_from_rules(None)
    mc.display_ascii()
    mc.state = mc.state.rotate_invert()
    mc.display_ascii()
    # mc.play_random()

