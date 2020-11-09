import numpy as np

def numpy_softmax(x):
    '''
        Basic numpy implementation of softmax over a 1-D vector.
    '''
    return np.exp(x) / sum(np.exp(x))