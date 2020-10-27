import numpy as np

def one_hot(argmax, length):
    '''
        Returns
        -------
        A numpy array of length `length` where the `argmax`-th item is 1, and the rest are 0.
    '''

    to_return = np.zeros(length)
    to_return[argmax] = 1

    return to_return