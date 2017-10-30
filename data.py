import numpy as np
import theano
import pickle
import gzip
import os
import re
from datasets.dl4mt.data_iterator import TextIterator


def _gen_data_add(seq_length, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    x_values = rng.uniform(low=0, high=1, size=seq_length)
    x_values = x_values.astype(dtype=theano.config.floatX)
    x_indicator = np.zeros(seq_length, dtype=np.bool)
    x_indicator[rng.randint(seq_length/2)] = 1
    x_indicator[rng.randint(seq_length/2, seq_length)] = 1
    #x = np.array(list(zip(x_values, x_indicator)))[np.newaxis]
    x = np.vstack((x_values, x_indicator)).T
    #y = np.sum(x_values[x_indicator], keepdims=True)/2.
    y = np.sum(x_values[x_indicator], keepdims=True)/2.
    return x, y
                        
                        
def gen_data_add(seq_length, batch_size, epoch_len, rng=None):
    x = np.zeros((batch_size, seq_length, 2), dtype=theano.config.floatX)
    y = np.zeros(batch_size, dtype=theano.config.floatX)
    
    def gen():
        for i in range(epoch_len):
            for b in range(batch_size):
                data = _gen_data_add(seq_length, rng=rng)
                x[b] = data[0]
                y[b] = data[1]
            yield x, y

    return gen

def one_hot(labels, num_classes=None):
    """
    Convert a unidimensional label array into a matrix of one-hot vectors
    -- takes only positive integer values (and zero).
    """
    
    if np.min(labels)<0:
        raise ValueError
    if num_classes==None:
        num_classes=np.max(labels)+1
    onehot_labels = np.zeros((len(labels), num_classes), dtype=np.int32)
    for i, l in enumerate(labels):
        onehot_labels[i, l] = 1
    return onehot_labels


def _gen_data_copy(seq_length, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    x = np.zeros(seq_length+20, dtype=theano.config.floatX)
    int_sequence = rng.randint(low=1, high=9, size=10)
    #x[:10] = (int_sequence - 4.5)/9.
    x[:10] = int_sequence
    #x -= x[:10].mean()
    #x /= x[:10].std()
    x[seq_length+9] = -1
    y_int = np.zeros(seq_length+20, dtype=np.int32)
    y_int[-10:] = int_sequence
    y = one_hot(y_int, num_classes=9)
    return x, y
        
        
def gen_data_copy(seq_length, batch_size, epoch_len, rng=None):
    x = np.zeros((batch_size, seq_length+20, 1), dtype=theano.config.floatX)
    y = np.zeros((batch_size, seq_length+20, 9), dtype=np.int32)
            
    def gen():
        for i in range(epoch_len):
            for b in range(batch_size):
                data = _gen_data_copy(seq_length, rng=rng)
                x[b,:,0] = data[0]
                y[b,:,:] = data[1]
            yield x, y

    return gen


def ptb_iterator(source, source_dict, batch_size, maxlen, char_level=False,
                 n_words_source=-1, rng=None):
    data = []
    if char_level:
        # Character level PTB
        if source.endswith('.gz'):
            source_file = gzip.open(source, 'r')
        else:
            source_file = open(source, 'r')
        
        # Make a dictionary mapping known characters to integers
        #   0 is 'unk'
        #   1 is 'end of sentence'
        # (48 entries)
        chars = ['<unk>', '\n', '#', '$', '&', "'", '*', '-', '.', '/', '\\',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'N', ' ',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        char_dict = dict(zip(chars, np.arange(len(chars))))
        
        # Make a list of all lines in integer encoded format
        
        for line in source_file:
            if len(line) > maxlen:
                continue
            encoded_line = []
            i = 0
            while i < len(line):
                ch = line[i]
                try:
                    encoded_line.append(char_dict[ch])
                except KeyError:
                    # Unknown characters are 0, including '<unk>'
                    encoded_line.append(0)
                    if line[i:i+5]=='<unk>':
                        i = i+4
                i += 1
            data.append(encoded_line)
            
    else:
        # Word level PTB
        text_iter = TextIterator(source=source,
                                source_dict=source_dict,
                                batch_size=batch_size,
                                maxlen=maxlen,
                                n_words_source=n_words_source)
        data = []
        for batch in text_iter:
            data.extend(batch)
            
    # Prepare data to sample batches from
    x_arr = np.zeros((len(data), maxlen), dtype=np.int32)
    m_arr = np.zeros((len(data), maxlen), dtype=np.uint8)
    y_arr = np.zeros((len(data), maxlen), dtype=np.int32)
    for i, line in enumerate(data):
        x_arr[i, 0:len(line)] = line
        m_arr[i, 0:len(line)+1] = 1
    y_arr[:,:-1] = x_arr[:,1:]
    
    if rng is None:
        rng = np.random.RandomState()
    
    num_batches = len(data)//batch_size
    if len(data)%batch_size:
        num_batches += 1
    
    def gen():
        indices = rng.permutation(len(data))
        for i in range(num_batches):
            x = x_arr[indices[i*batch_size:(i+1)*batch_size]]
            m = m_arr[indices[i*batch_size:(i+1)*batch_size]]
            y = y_arr[indices[i*batch_size:(i+1)*batch_size]]
            yield x, m, y
            
    return gen
    

def mnist_iterator(data, batch_size, permute=0, permutation=None,
                   new_shape=None, shuffle=False, rng=None):
    '''
    permute=0 : do not permute
    permute=1 : permute once
    permute=2 : randomly permute every sample
    
    new_shape : rescale images to this size (no rescaling if new_shape is None)
    shuffle   : shuffle the data samples
    
    Note: When permuting, use the rng_seed to make all iterators permute
          in the same order.
    '''
    x, y = data
    assert(len(x)==len(y))
    num_batches = len(x)//batch_size
    if len(x)%batch_size:
        num_batches += 1
        
    # total number of pixels in an image
    vec_len = 28**2
    if new_shape is not None:
        vec_len = np.prod(new_shape)
        # numpy permutation returns float64 when the input is uint64 (bug)
        vec_len = vec_len.astype(np.int64)
        
    # RNG
    if rng is None:
        rng = np.random.RandomState()
    
    # predetermined permutation
    constant_permutation = permutation
    if constant_permutation is None:
        constant_permutation = rng.permutation(vec_len)        
    
    # rescale to new_shape if needed
    def rescale(image_stack):
        from scipy.misc import imresize
        new_stack = []
        for img in image_stack:
            new_img = imresize(img.reshape((28,28)),
                               new_shape,
                               interp='bilinear')
            new_stack.append(new_img.flatten())
        return np.array(new_stack)/255.
    
    # return this generator
    def gen():
        for b in range(num_batches):
            if shuffle:
                indices = rng.permutation(len(x))
            else:
                indices = np.arange(len(x))
            x_batch = x[indices[b*batch_size:(b+1)*batch_size]]
            if new_shape is not None:
                x_batch = rescale(x_batch)
            x_batch = np.expand_dims(x_batch, 2).astype(np.float32)
            y_batch = y[indices[b*batch_size:(b+1)*batch_size]]
            y_batch = y_batch.astype(np.int32)
            if permute==1:
                x_batch = x_batch[:,constant_permutation]
            if permute==2:
                for i in range(len(x_batch)):
                    new_permutation = rng.permutation(vec_len)
                    x_batch[i,:] = x_batch[i,new_permutation]
            yield x_batch, y_batch
    
    return gen

