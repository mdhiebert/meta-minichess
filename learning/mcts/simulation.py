import logging
import random
from minichess.games.abstract.board import AbstractBoardStatus
from minichess.games.abstract.piece import PieceColor


class Simulator:
    def __init__(self, env_class, board, active_color, simulation_iterations=75):
        self.env = env_class()
        self.env.board = board
        self.active_color = active_color
        self.simulation_iterations = simulation_iterations

    def rollout_to_completion(self):
        '''
            Simulates this game with random moves to completion, or the simulation iteration count, whichever comes first.

            Returns
            -------
            terminal status, reward
        '''

        sim_iter = 0
        reward = 0
        done = False

        while not done and sim_iter < self.simulation_iterations:
            actions = self.env.legal_actions()

            if len(actions) == 0:
                done = True
                break

            action = random.choice(actions)

            _, reward, done, _ = self.env.step(action)

            sim_iter += 1


        if sim_iter >= self.simulation_iterations:
            logging.debug('Cut simulation at iteration {}. Forced Result: {}.'.format(sim_iter, AbstractBoardStatus.DRAW))
            return AbstractBoardStatus.DRAW, reward

        logging.debug('Finished simulation at iteration {}. Result: {}.'.format(sim_iter, self.env.board.status))
        return self.env.board.status, reward


