import logging
from learning.mcts.simulation import Simulator
import random
import json

from minichess.games.abstract.board import AbstractChessBoard, AbstractBoardStatus
from minichess.games.abstract.piece import PieceColor

# EMPTY_NODE = {
#     'white': {
#         'num': 0,
#         'den': 0,
#         'max_reward': 0
#     },
#     'black': {
#         'num': 0,
#         'den': 0,
#         'max_reward': 0
#     }
# }

def EMPTY_NODE():
    d = dict()
    d['white'] = dict()
    d['white']['num'] = 0
    d['white']['den'] = 0
    d['white']['max_reward'] = 0
    d['black'] = dict()
    d['black']['num'] = 0
    d['black']['den'] = 0
    d['black']['max_reward'] = 0
    return d

class MCTS:
    '''
        TODO better docstring

        https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

        Monte Carlo Tree Search

        The focus of MCTS is on the analysis of the most promising moves, expanding the search tree based on random sampling of the search space. The 
        application of Monte Carlo tree search in games is based on many playouts, also called roll-outs. In each playout, the game is played out to 
        the very end by selecting moves at random. The final game result of each playout is then used to weight the nodes in the game tree so that 
        better nodes are more likely to be chosen in future playouts.

        The most basic way to use playouts is to apply the same number of playouts after each legal move of the current player, then choose the move 
        which led to the most victories.[10] The efficiency of this method—called Pure Monte Carlo Game Search—often increases with time as more playouts 
        are assigned to the moves that have frequently resulted in the current player's victory according to previous playouts. Each round of Monte Carlo 
        tree search consists of four steps:[35]

        Selection: Start from root R and select successive child nodes until a leaf node L is reached. The root is the current game state and a leaf 
        is any node that has a potential child from which no simulation (playout) has yet been initiated. The section below says more about a way of 
        biasing choice of child nodes that lets the game tree expand towards the most promising moves, which is the essence of Monte Carlo tree search.
        
        Expansion: Unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child nodes and choose node C from 
        one of them. Child nodes are any valid moves from the game position defined by L.
        
        Simulation: Complete one random playout from node C. This step is sometimes also called playout or rollout. A playout may be as simple as choosing 
        uniform random moves until the game is decided (for example in chess, the game is won, lost, or drawn).
        
        Backpropagation: Use the result of the playout to update information in the nodes on the path from C to R.
    '''
    def __init__(self, env_class, state_dict: dict = None):
        self.state_dict = dict() if state_dict == None else state_dict
        self.env_class = env_class

    def save_to_json(self, save_file = 'learning/mcts/mcts_data.json'):
        '''
            Converts this MCTS to JSON.

            Returns
            -------
            str representing this MCTS in JSON format.
        '''
        with open(save_file, 'w') as outfile:
            json.dump(self.state_dict, outfile)

    @staticmethod
    def from_json_file(env_class, json_filename: str = 'learning/mcts/mcts_data.json'):
        '''
            Creates a MCTS instance from a JSON string.

            Parameters
            ----------
            json_filename :: str : A filename of json with following format:
                {
                    "state_hash_1": {
                        "white": {
                            "num": x,
                            "den": y,
                            "max_reward": z
                        },
                        "black": {
                            "num": x,
                            "den": y,
                            "max_reward": z
                        }
                    }
                    ...
                }
        '''
        with open(json_filename, 'r') as f:
            state_dict = json.load(f)

        return MCTS(env_class, state_dict)

    def iterate(self, root_env, currently_active_color: PieceColor):
        '''
            Conduct one full Monte Carlo Tree Search iteration with `root_state` as our root node.

            Parameters
            ----------
            root_state :: gym Env : the current (root) state of the minichess game

            currently_active_color :: PieceColor : the currently active color of the minichess game

            Returns
            -------
            True if successful, else False
        '''

        # select a leaf
        logging.debug('Selecting leaf...')
        leaf_state, leaf_color, seen = self.selection(root_env)

        # add our leaf to seen
        logging.debug('Adding leaf to seen...')
        seen.append((str(hash(leaf_state)), leaf_color))

        # expand it out one step
        logging.debug('Expanding leaf node one step...')
        expanded_state, expanded_color = self.expansion(leaf_state, leaf_color)
        expanded_color = leaf_color.invert()


        # simulate to termination
        terminal_result, reward = self.simulation(expanded_state, expanded_color)

        return self.backpropagation(expanded_state, expanded_color, seen, terminal_result, reward)


    def selection(self, env):
        '''
            With `(state, currently_active_color)` as the root node, perform a random
            search down known nodes until we arrive at a leaf node.

            We define our a leaf node as any node for which the state hash does not yet
            exist in our `state_dict` OR any node for which `state_dict[currently_active_color]['den']` is 0.

            Parameters
            ----------
            state :: MiniChessState : the current state of the board, acting as the root of our tree from which to search

            currently_active_color :: PieceColor : the currently active color in our game state

            Returns
            -------
            tuple of (state, currently_active_color, seen) where seen is a list of (state_hash, color)
            for all parent nodes we encountered along the way
        '''

        seen = []

        cac = str(env.board.active_color)

        while str(hash(env)) in self.state_dict and self.state_dict[str(hash(env))][cac]['den'] != 0:

            children = env.legal_actions()
            
            if len(children) == 0: return (env, env.board.active_color, seen) # terminal state
            
            seen.append((str(hash(env)), env.board.active_color))

            child = random.choice(children)
            _ = env.step(child)

            cac = str(env.board.active_color)

        # at this point we have reached a leaf node
        return (env, env.board.active_color, seen)

    def expansion(self, leaf_env: AbstractChessBoard, leaf_color: PieceColor):
        '''
            Expand our selected leaf node out one iteration and return the expanded node.

            Parameters
            ----------
            leaf_state :: AbstractChessBoard : the current state of the board, acting as the root of our tree from which to search

            leaf_color :: PieceColor : the currently active color in our game state

            Returns
            -------
            the expanded state as a AbstractChessBoard if there is one, else None
        '''
        
        # add to our state_dict if it does not yet exist
        if str(hash(leaf_env)) not in self.state_dict:
            self.state_dict[str(hash(leaf_env))] = EMPTY_NODE()
        
        # generate possible children
        children = leaf_env.legal_actions()

        # there are no children, perhaps because leaf_state is a terminal state
        if len(children) == 0:
            logging.debug('Leaf node is terminal.')
            return leaf_env, leaf_color

        child = random.choice(children)

        leaf_env.step(child)

        return leaf_env, leaf_color.invert()

    def simulation(self, expanded_env: AbstractChessBoard, expanded_color: PieceColor):
        '''
            Simulate a chess game from this state to termination.

            Parameters
            ----------
            expanded_state :: AbstractChessBoard : the state of our expanded node

            expanded_color :: PieceColor : the currently active color in leaf game state

            Returns
            -------
            tuple of (AbstractChessStatus of the result of this game, reward)
        '''

        sim = Simulator(self.env_class, expanded_env.board, expanded_color)

        return sim.rollout_to_completion()

    def backpropagation(self, expanded_state: AbstractChessBoard, expanded_color, seen: list, terminal_result: AbstractBoardStatus, reward):
        '''
            Backpropagate the results of a simulation back up the tree, incrementing both numerator 
            and denominator in our `state_dict` as follows:

            if winner color equals color of this node: num += 1
            elif draw: num += 1
            else: no change to num

            den += 1 for all

            Parameters
            ----------
            expanded_state :: MiniChessState : the state of our expanded node

            expanded_color :: PieceColor : the currently active color in expanded game state

            seen :: tuple(hashcodes, colors) : the list of seen nodes in our path

            reward :: float : the reward of this simulation

            Return
            ------
            True if successful, else False
        '''

        # handle last node
        self._backprop_node(str(hash(expanded_state)), expanded_color, terminal_result, reward)

        # handle parent nodes
        for node_hash, color in seen:
            self._backprop_node(node_hash, color, terminal_result, reward)

        return True

    def _backprop_node(self, node_hashcode: str, color: PieceColor, result: AbstractBoardStatus, reward: float):

        logging.debug('Handling backprob for node with hash {}'.format(node_hashcode))

        if node_hashcode not in self.state_dict:
            self.state_dict[node_hashcode] = EMPTY_NODE()

        node_dict = self.state_dict[node_hashcode][str(color)]

        if result == AbstractBoardStatus.DRAW:
            self._increment_node(node_dict, 0.5)
        elif result == AbstractBoardStatus.WHITE_WIN:
            self._increment_node(node_dict, 1 if color == PieceColor.WHITE else 0)
        elif result == AbstractBoardStatus.BLACK_WIN:
            self._increment_node(node_dict, 1 if color == PieceColor.BLACK else 0)
        else:
            raise RuntimeError('Expected terminal state for MCTS backpropagation but got {}'.format(result.name))  

        node_dict['max_reward'] = max(node_dict['max_reward'], reward) if color == PieceColor.WHITE else min(node_dict['max_reward'], reward)

    def _increment_node(self, node_dict: dict, num_amount: float, den_amount: int = 1):
        '''
            Increments `node_dict`'s `num` and `den` by `num_amount` and `den_amount`, respectively.

            Parameters
            ----------
            node_dict :: dict : the dictionary of this node at specific color, accessed by `self.state_dict[HASH][COLOR]

            num_amount :: float : the amount to increment `num` by

            den_amount :: int : the amount to increment `den` by
        '''
        node_dict['num'] += num_amount
        node_dict['den'] += den_amount

    def suggest_move(self, current_state: AbstractChessBoard, active_color: PieceColor):
        '''
            Suggest a move that is most conducive to success

            Parameters
            ----------
            current_state :: AbstractChessBoard : the current state of the board

            active_color :: PieceColor : the currently active color

            Returns
            -------
            AbstractChessMove
        '''
        # invert color
        color = str(active_color.invert())

        # get next states and filter out un-simulated next_states
        next_moves = current_state.possible_moves(active_color)
        # this is a beefy listcomp and could be split up. TODO
        next_nodes = [(move, str(hash(current_state.apply_move(move)))) for move in next_moves if str(hash(current_state.apply_move(move))) in self.state_dict and self.state_dict[str(hash(current_state.apply_move(move)))][color]['den'] != 0]

        # if there are no next_nodes, return random
        if len(next_nodes) == 0:
            return random.choice(next_moves)

        # our heuristic is to choose the minimum weight of the next step (from the opposite perspective)
        # low black success rate -> high white success rate
        best_move = sorted(next_nodes, key=lambda x: self.state_dict[x[1]][color]['num'] / self.state_dict[x[1]][color]['den'])[0][0]

        return best_move




