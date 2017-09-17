import lasagne
import numpy as np

class Identity(lasagne.init.Initializer):
    def sample(self, shape):
        if len(shape) != 2:
            raise ValueError("Identity initialization can only be applied to "
                             "2-dimensional matrices. Error attemptint to "
                             "apply it to a {}-dimensional "
                             "input.".format(len(shape)))
        return lasagne.utils.floatX(np.eye(*shape))

class Orthogonal(lasagne.init.Initializer):
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.gain = gain

    def sample(self, shape):
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are "
                               "supported.")

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = lasagne.random.get_rng().normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = np.dot(u, v)
        return lasagne.utils.floatX(self.gain * q)
