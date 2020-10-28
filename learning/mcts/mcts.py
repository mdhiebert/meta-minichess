from minichess.state import MiniChessState
from minichess.pieces import PieceColor
from minichess.minichess import MiniChess, TerminalStatus
import random
import json

EMPTY_NODE = {
    'white': {
        'num': 0,
        'den': 0
    },
    'black': {
        'num': 0,
        'den': 0
    }
}

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
    def __init__(self, state_dict: dict = None):
        self.state_dict = dict() if state_dict == None else state_dict

    def json(self, save_file = 'learning/mcts/mcts_data.json'):
        '''
            Converts this MCTS to JSON.

            Returns
            -------
            str representing this MCTS in JSON format.
        '''
        with open(save_file, 'w') as outfile:
            json.dump(self.state_dict, outfile)

    @staticmethod
    def from_json(self, json_filename: str = 'learning/mcts/mcts_data.json'):
        '''
            Creates a MCTS instance from a JSON string.

            Parameters
            ----------
            json_filename :: str : A filename of json with following format:
                {
                    "state_hash_1": {
                        "white": {
                            "num": x,
                            "den": y
                        },
                        "black": {
                            "num": x,
                            "den": y
                        }
                    }
                    ...
                }
        '''
        with open(json_filename, 'r') as f:
            state_dict = json.load(f)

    def iterate(self, root_state: MiniChessState, currently_active_color: PieceColor):
        '''
            Conduct one full Monte Carlo Tree Search iteration with `root_state` as our root node.

            Parameters
            ----------
            root_state :: MiniChessState : the current (root) state of the minichess game

            currently_active_color :: PieceColor : the currently active color of the minichess game

            Returns
            -------
            True if successful, else False
        '''

        # select a leaf
        leaf_state, leaf_color, seen = self.selection(root_state, currently_active_color)

        # add our leaf to seen
        seen.append((str(hash(leaf_state)), leaf_color))

        # expand it out one step
        expanded_state, expanded_color = self.expansion(leaf_state, leaf_color)
        expanded_color = PieceColor.invert(leaf_color)


        # simulate to termination
        terminal_result = self.simulation(expanded_state, expanded_color)

        return self.backpropagation(expanded_state, expanded_color, seen, terminal_result)


    def selection(self, state: MiniChessState, currently_active_color: PieceColor):
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

        cac = PieceColor.to_string(currently_active_color)

        while str(hash(state)) in self.state_dict and self.state_dict[str(hash(state))][cac]['den'] != 0:

            children = state.possible_next_states(currently_active_color)
            
            if len(children) == 0: return (state, currently_active_color, seen) # terminal state
            
            seen.append((str(hash(state)), currently_active_color))

            state = random.choice(children)
            currently_active_color = PieceColor.invert(currently_active_color)
            cac = PieceColor.to_string(currently_active_color)

        # at this point we have reached a leaf node
        return (state, currently_active_color, seen)

    def expansion(self, leaf_state: MiniChessState, leaf_color: PieceColor):
        '''
            Expand our selected leaf node out one iteration and return the expanded node.

            Parameters
            ----------
            leaf_state :: MiniChessState : the current state of the board, acting as the root of our tree from which to search

            leaf_color :: PieceColor : the currently active color in our game state

            Returns
            -------
            the expanded state as a MiniChessState if there is one, else None
        '''
        
        # add to our state_dict if it does not yet exist
        if str(hash(leaf_state)) not in self.state_dict:
            self.state_dict[str(hash(leaf_state))] = EMPTY_NODE
        
        # generate possible children
        children = leaf_state.possible_next_states(leaf_color)

        # there are no children, perhaps because leaf_state is a terminal state
        if len(children) == 0: return leaf_state, leaf_color

        expanded_state = random.choice(children)

        return expanded_state, PieceColor.invert(leaf_color)

    def simulation(self, expanded_state: MiniChessState, expanded_color: PieceColor):
        '''
            Simulate a chess game from this state to termination.

            Parameters
            ----------
            expanded_state :: MiniChessState : the state of our expanded node

            expanded_color :: PieceColor : the currently active color in leaf game state

            Returns
            -------
            TerminalStatus of the result of this game.
        '''

        mc = MiniChess.init_from_state(expanded_state, expanded_color)

        return mc.rollout_to_completion()

    def backpropagation(self, expanded_state: MiniChessState, expanded_color, seen: list, terminal_result: TerminalStatus):
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

            Return
            ------
            True if successful, else False
        '''

        # handle last node
        self._backprop_node(str(hash(expanded_state)), expanded_color, terminal_result)

        # handle parent nodes
        for node_hash, color in seen:
            self._backprop_node(node_hash, color, terminal_result)

        return True

    def _backprop_node(self, node_hashcode: str, color: PieceColor, result: TerminalStatus):

        if node_hashcode not in self.state_dict:
            self.state_dict[node_hashcode] = EMPTY_NODE

        node_dict = self.state_dict[node_hashcode][PieceColor.to_string(color)]

        if result == TerminalStatus.DRAW:
            self._increment_node(node_dict, 0.5)
        elif result == TerminalStatus.WHITE_WIN:
            self._increment_node(node_dict, 1 if color == PieceColor.WHITE else 0)
        elif result == TerminalStatus.BLACK_WIN:
            self._increment_node(node_dict, 1 if color == PieceColor.BLACK else 0)
        else:
            raise RuntimeError('Expected terminal state for MCTS backpropagation but got {}'.format(result.name))  

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

    def suggest_move(self, current_state: MiniChessState, active_color: PieceColor):
        '''
            Suggest a move that is most conducive to success

            Parameters
            ----------
            current_state :: MiniChessState : the current state of the board

            active_color :: PieceColor : the currently active color

            Returns
            -------
            MiniChessMove
        '''
        # invert color
        color = PieceColor.to_string(PieceColor.invert(active_color))

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




