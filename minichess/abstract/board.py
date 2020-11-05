from minichess.abstract.action import AbstractChessAction

class AbstractChessBoard:
    '''
        An abstract data type representing a Chess Board,
        agnostic of rules.
    '''
    def __init__(self, side_length):
        self.side_length = side_length
        self.move_history = []

    def push(self, action: AbstractChessAction):
        '''
            Push a `AbstractChessAction` to this chess board,
            applying it to the current board and updating
            current state appropriately.

            Parameters
            ----------
            action :: AbstractChessAction : an action to apply to
            this game board.
        '''
        raise NotImplementedError

    def pop(self) -> AbstractChessAction:
        '''
            Removes the most recent move from this board, undoing
            its effects and returning the move itself.

            Returns
            -------
            An AbstractChessAction for the most recent action pushed
            to this Board.
        '''
        raise NotImplementedError