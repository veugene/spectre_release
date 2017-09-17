from theano import tensor as T

def orthogonality(x):
    '''
    Penalty for deviation from orthogonality:
    
    ||dot(x.T, x) - I||**2
    '''
    xTx = T.dot(x.T, x)
    return T.sum(T.square(xTx - T.identity_like(xTx)))

def constant_l2(val):
    def penalty(x):
        '''
        Sum of squares of differences between elements of x and 'val'
        
        sqrt((x-1)**2)
        '''
        return T.sum(T.square(x - val*T.ones_like(x)))
    return penalty
    
def constant_l1(val):
    def penalty(x):
        '''
        Sum of magnitudes of differences between elements of x and 'val'
        
        sqrt((x-1)**2)
        '''
        return T.sum(T.abs(x - val*T.ones_like(x)))
    return penalty
    
