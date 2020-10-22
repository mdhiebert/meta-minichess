from minichess.minichess import MiniChess
from minichess.pieces import PieceColor

# TODO check / checkmate / terminal states
# TODO point values
# TODO generating action options

if __name__ == "__main__":
    mc = MiniChess.init_from_rules(None)
    mc.display_ascii()
    print('+-----------------+')
    nxt = mc.immediate_states()
    # print(str(nxt))
    # print('+-----------------+')

    for s in nxt:
        print(str(s))
        print('+-----------------+')

    # for i in range(2):
    #     nxt = nxt.possible_next_states(PieceColor.WHITE)[1]
    #     print(str(nxt))
    #     print('+-----------------+')

