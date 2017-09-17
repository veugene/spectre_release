from theano import tensor as T

def oplu(x):
    z = x.copy()
    
    # Pair up the neurons (each element in a with each in b)
    a = z[:,  ::2]
    b = z[:, 1::2]
    
    # Find elments to swap (want pairs of {max, min}; swap min)
    min_ind = (a<b).nonzero()
    
    # Swap
    T.inc_subtensor(a[min_ind],  b[min_ind])
    T.inc_subtensor(b[min_ind], -a[min_ind])
    T.inc_subtensor(a[min_ind],  b[min_ind])
    T.set_subtensor(b[min_ind], -b[min_ind])
    
    return z 

def piecewise_relu(m):
    def f(x):
        lt_zero = (x<0).nonzero()
        gt_zero = (x>=0).nonzero()
        out = T.zeros_like(x)
        out[lt_zero] = x[lt_zero]*m
        out[gt_zero] = x[gt_zero]
        return out
    return f
