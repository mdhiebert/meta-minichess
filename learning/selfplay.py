from minichess.minichess import MiniChess, TerminalStatus
from learning.action import MiniChessAction
from learning.model import MiniChessModel
import torch
import numpy as np

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    counter = 0

    mc = MiniChess()
    player = MiniChessModel().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(player.parameters(), lr=1e-3)

    mc.display_ascii()

    while mc.terminal_status() == TerminalStatus.ONGOING:

        if counter == 3: break

        # get the board in vector form
        vector = torch.from_numpy(mc.current_state().vector()).to(device)

        # feed it through our player model and get a move prediction
        piece_vector, move_vector, magnitude_vector = player(vector)
        action = MiniChessAction.from_vectors(mc.active_color, piece_vector.cpu().detach().numpy(), 
                                                move_vector.cpu().detach().numpy(), 
                                                magnitude_vector.cpu().detach().numpy())

        # loss calculations

        while not action.is_valid_action(mc.current_state()): # the model is trying to make an invalid move

            # pick a random valid move
            legal_move = np.random.choice(mc.current_state().possible_moves(mc.active_color))
            legal_action =  MiniChessAction.from_move(legal_move)
            act_piece_vector, act_move_vector, act_magnitude_vector = [torch.from_numpy(x).to(device) for x in legal_action.vectors()]

            loss = loss_fn(piece_vector, act_piece_vector) + loss_fn(move_vector, act_move_vector) + loss_fn(magnitude_vector, act_magnitude_vector)

            # penalize heavily
            opt.zero_grad()
            loss.backward()
            opt.step()

            # try again
            piece_vector, move_vector, magnitude_vector = player(vector)
            action = MiniChessAction.from_vectors(mc.active_color, piece_vector, move_vector, magnitude_vector)

        # our real loss comparison will come from MCTS
        loss = loss_fn(piece_vector, piece_vector) # TODO
        
        opt.zero_grad()
        loss.backward()

        mc.apply_action(action)

        mc.display_ascii()

        counter += 1


        

