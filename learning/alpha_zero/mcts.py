import logging
import math
from minichess.games.abstract.piece import PieceColor
from minichess.games.gardner.action import GardnerChessAction

import numpy as np

from minichess.games.abstract.board import AbstractBoardStatus, AbstractChessBoard
from minichess.games.gardner.board import LEN_ACTION_SPACE

EPS = 1e-8

log = logging.getLogger(__name__)

# greatly borrowed from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py

class MCTS():
    """
    This class handles the MCTS tree.

    Parameters
    ----------
    game :: AbstractChessBoard : the game to conduct searches on

    Action :: type(AbstractChessAction) : the Action class appropriate for the game

    net :: pytorch network : the player network

    args :: dict : parameters/arguments to pass in
    """

    def __init__(self, game, Action, net, args):
        self.game = game
        self.Action = Action
        self.net = net
        self.args = args

        # initialize values
        self.Qsa = {} # stores Q values for (s,a)
        self.Nsa = {} # stores times edge (s,a) was visited
        self.Ns  = {} # stores times s was visited
        self.Ps  = {} # stores initial policy (from net)

        self.Es  = {} # stores game.getGameEnded for board s
        self.Vs  = {} # stores game.getValidMoves for board s

    def getActionProb(self, board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        the canonical version of the current board.

        Parameters
        ----------
        board :: minichess.AbstractChessBoard : the board to calculate action probabilities for.

        Returns
        -------
        probs: a policy vector where the probability of the ith action is
                proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(board.copy())

        # TODO below

        s = str(board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(LEN_ACTION_SPACE)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board: AbstractChessBoard, active_color=PieceColor.WHITE):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Parameters
        ----------
        board :: AbstractChessBoard : the board to search from. We assume that the active color
            of the board is correct when it is passed in.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = str(board)

        board.active_color = active_color

        # determine terminal-ness of this node
        if s not in self.Es:
            if board.status == AbstractBoardStatus.WHITE_WIN:
                self.Es[s] = 1
            elif board.status == AbstractBoardStatus.BLACK_WIN:
                self.Es[s] = -1
            elif board.status == AbstractBoardStatus.DRAW:
                self.Es[s] = 0.1
            else: # ONGOING
                self.Es[s] = 0

        # not ongoing -> terminal
        if self.Es[s] != 0: 
            return -self.Es[s]

        # handle predictions
        if s not in self.Ps:
            # leaf node

            # set prediction
            self.Ps[s], v = self.net.predict(board.state_vector())

            # get valid move mask
            valids = board.legal_action_mask()

            # mask invalid moves
            self.Ps[s] = self.Ps[s] * valids

            # renormalize post-mask
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            # set valid mask
            self.Vs[s] = valids

            # set visit count
            self.Ns[s] = 0
            return -v


        # get valids
        valids = self.Vs[s]

        print(board, valids.nonzero()[0])

        # set minimums
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(LEN_ACTION_SPACE):

            # if this is a valid move
            if valids[a]:
                
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        board.push(self.Action.decode(a, board))
        next_s = board.copy()

        v = self.search(next_s.copy(), active_color=active_color.invert())

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
