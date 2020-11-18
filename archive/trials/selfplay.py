from minichess.minichess import MiniChess, TerminalStatus
from minichess.pieces import PieceColor
from learning.action import MiniChessAction
from learning.model import MiniChessModel
from learning.muzero.mcts.mcts import MCTS
import torch
import numpy as np
import matplotlib.pyplot as plt

MCTS_FILE = ''
MCTS_SIMS_PER_STEP = 0
GAMES = 5
COUNTER_BREAK = 100
DEBUG = False

# Self-Play Training Module

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # init our two players
    
    white_player = MiniChessModel().to(device)
    black_player = MiniChessModel().to(device)

    # loss functions
    loss_fn = torch.nn.NLLLoss()
    white_opt = torch.optim.Adam(white_player.parameters(), lr=1e-3)
    black_opt = torch.optim.Adam(black_player.parameters(), lr=1e-3)

    # init our MCTS

    mcts = MCTS.from_json(MCTS_FILE) if MCTS_FILE else MCTS()

    incorrects = []

    for game_num in range(GAMES):

        mc = MiniChess()
        if DEBUG: mc.display_ascii()
        counter = 0

        incorrect_move_counter = 0

        # game loop
        while mc.terminal_status() == TerminalStatus.ONGOING:

            if counter >= COUNTER_BREAK: break

            player = white_player if mc.active_color == PieceColor.WHITE else black_player
            opt = white_opt if mc.active_color == PieceColor.WHITE else black_opt

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


            

            while not action.is_valid_action(mc.current_state()): # the model is trying to make an invalid move
                incorrect_move_counter += 1

                # pick a random valid move
                legal_moves = mc.current_state().possible_moves(mc.active_color, filter_by_check=True)
                legal_move = np.random.choice(legal_moves)
                legal_action =  MiniChessAction.from_move(legal_move)

                # convert that random legal action to a vector
                act_piece_vector, act_move_vector, act_magnitude_vector = [torch.from_numpy(x).to(device) for x in legal_action.vectors()]
                act_piece_argmax = torch.argmax(act_piece_vector).unsqueeze(0)
                act_move_argmax = torch.argmax(act_move_vector).unsqueeze(0)
                act_magnitude_argmax = torch.argmax(act_magnitude_vector).unsqueeze(0)

                # loss to learn rules!
                loss = loss_fn(piece_vector, act_piece_argmax) + loss_fn(move_vector, act_move_argmax) + loss_fn(magnitude_vector, act_magnitude_argmax)

                # penalize heavily
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


            # Conduct MCTS iterations
            for _ in range(MCTS_SIMS_PER_STEP):
                mcts.iterate(mc.current_state(), mc.active_color)

            # Get the relatively best move from our MCTS
            best_move = mcts.suggest_move(mc.current_state(), mc.active_color)
            best_action = MiniChessAction.from_move(best_move)

            # convert best action to vectors
            best_piece_vector, best_move_vector, best_magnitude_vector = [torch.from_numpy(x).to(device) for x in best_action.vectors()]
            best_piece_argmax = torch.argmax(best_piece_vector).unsqueeze(0)
            best_move_argmax = torch.argmax(best_move_vector).unsqueeze(0)
            best_magnitude_argmax = torch.argmax(best_magnitude_vector).unsqueeze(0)

            # our real loss comparison will come from MCTS 
            loss =  loss_fn(piece_vector, best_piece_argmax) + loss_fn(move_vector, best_move_argmax) + loss_fn(magnitude_vector, best_magnitude_argmax)
            opt.zero_grad()
            loss.backward()
            opt.step()

            mc.apply_action(action)

            if DEBUG: mc.display_ascii()

            counter += 1

        incorrects.append(incorrect_move_counter)
        if DEBUG: print('{} incorrect moves!'.format(incorrect_move_counter))
        print(game_num)
    mcts.json()

    if DEBUG: print(mc.terminal_status())

    plt.plot(incorrects)
    print(incorrects)

    plt.show()




        

