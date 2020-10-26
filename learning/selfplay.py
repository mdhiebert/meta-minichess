from minichess.minichess import MiniChess, TerminalStatus
from minichess.pieces import PieceColor
from learning.action import MiniChessAction
from learning.model import MiniChessModel
import torch
import numpy as np

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    counter = 0

    mc = MiniChess()
    white_player = MiniChessModel().to(device)
    black_player = MiniChessModel().to(device)

    loss_fn = torch.nn.NLLLoss()
    white_opt = torch.optim.Adam(white_player.parameters(), lr=1e-3)
    black_opt = torch.optim.Adam(black_player.parameters(), lr=1e-3)

    mc.display_ascii()

    while mc.terminal_status() == TerminalStatus.ONGOING:

        print(mc.active_color)

        player = white_player if mc.active_color == PieceColor.WHITE else black_player
        opt = white_opt if mc.active_color == PieceColor.WHITE else black_opt

        # if counter == 10: break

        # get the board in vector form
        vector = torch.from_numpy(mc.current_state().vector()).to(device)

        # feed it through our player model and get a move prediction
        piece_vector, move_vector, magnitude_vector = player(vector)
        piece_vector = piece_vector.unsqueeze(0)
        move_vector = move_vector.unsqueeze(0)
        magnitude_vector = magnitude_vector.unsqueeze(0)
        action = MiniChessAction.from_vectors(mc.active_color, piece_vector.cpu().detach().numpy(), 
                                                move_vector.cpu().detach().numpy(), 
                                                magnitude_vector.cpu().detach().numpy())

        # loss calculations

        c = 0

        while not action.is_valid_action(mc.current_state()): # the model is trying to make an invalid move
            # pick a random valid move
            legal_moves = mc.current_state().possible_moves(mc.active_color, filter_by_check=True)
            legal_move = np.random.choice(legal_moves)
            legal_action =  MiniChessAction.from_move(legal_move)

            act_piece_vector, act_move_vector, act_magnitude_vector = [torch.from_numpy(x).to(device) for x in legal_action.vectors()]
            act_piece_argmax = torch.argmax(act_piece_vector).unsqueeze(0)
            act_move_argmax = torch.argmax(act_move_vector).unsqueeze(0)
            act_magnitude_argmax = torch.argmax(act_magnitude_vector).unsqueeze(0)

            loss = loss_fn(piece_vector, act_piece_argmax) + loss_fn(move_vector, act_move_argmax) + loss_fn(magnitude_vector, act_magnitude_argmax)

            # print('suggested', action)
            # print('actual', legal_action)

            # penalize heavily
            loss *= 10
            opt.zero_grad()
            loss.backward()
            opt.step()

            # try again
            piece_vector, move_vector, magnitude_vector = player(vector)
            piece_vector = piece_vector.unsqueeze(0)
            move_vector = move_vector.unsqueeze(0)
            magnitude_vector = magnitude_vector.unsqueeze(0)
            action = MiniChessAction.from_vectors(mc.active_color, piece_vector.cpu().detach().numpy(), 
                                                    move_vector.cpu().detach().numpy(), 
                                                    magnitude_vector.cpu().detach().numpy())
            # print('revised', action)

        # our real loss comparison will come from MCTS 
        loss = loss_fn(piece_vector, torch.argmax(piece_vector).unsqueeze(0)) # TODO
        opt.zero_grad()
        loss.backward()
        opt.step()

        mc.apply_action(action)

        mc.display_ascii()

        counter += 1

    print(mc.terminal_status())


        

