from lasagne.updates import get_or_compute_grads
from collections import OrderedDict
from theano import tensor as T
import theano
import numpy as np

def geoSGD(loss_or_grads, params, learning_rate):
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    lr = learning_rate

    for param, grad in zip(params, grads):
        W = param.get_value(borrow=True)
        G = grad
        A = T.dot(G.T, W) - T.dot(W.T, G)
        I = T.identity_like(A)
        cayley = T.dot(T.nlinalg.matrix_inverse(I+(lr/2.)*A), I-(lr/2.)*A)
        updates[param] = T.dot(cayley, W)

    return updates
