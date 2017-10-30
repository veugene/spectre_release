#!/usr/bin/env python

# System libs
from __future__ import print_function
if hasattr(__builtins__, 'raw_input'):
    input = raw_input
from collections import OrderedDict
import os
import sys
import shutil
import gzip
import pickle
import copy
import argparse

# Setup matplotlib renderer to avoid using the screen (must be done early)
import matplotlib
matplotlib.use('Agg')

# Installed libs
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import lasagne
import h5py

# Local
from data import (gen_data_add,
                  gen_data_copy,
                  ptb_iterator,
                  mnist_iterator)
from custom_rnn import (RecurrentLayer,
                        FactorizedRecurrentLayer)
import init
import penalties
import nonlinearities as custom_nonlinearities
from geodesic_sgd import geoSGD




def parse():
    parser = argparse.ArgumentParser(description="Train an rnn.")
    
    # model_params
    g1 = parser.add_argument_group("model_params")
    g1.add_argument('--model_type',
                    help="type of RNN to train",
                    required=False, type=str, default='factorized_rnn',
                    choices=['simple_rnn', 'factorized_rnn'])
    g1.add_argument('--num_units',
                    help="size of the hidden representation",
                    required=False, type=int, default=128)
    g1.add_argument('--margin',
                    help="the maximum allowed deviation (margin) of singular "
                         "values from 1 (from orthogonality)",
                    required=False, type=float, default=0.)
    g1.add_argument('--bias',
                    help="bias term applied to all linear operations",
                    required=False, type=float, default=0.)
    g1.add_argument('--W_hid_to_hid',
                    help="initialization method for the hidden-to-hidden "
                         "transition matrix, specified by the initializer "
                         "name from lasagne.init or the custom init.py",
                    required=False, type=str, default='Orthogonal')
    g1_1 = parser.add_argument_group('nonlinearity',
                                     "hidden-to-hidden transition nonlinearity")
    g1_1.add_argument('--nonlinearity',
                      help="hidden-to-hidden transition nonlinearity, "
                           "specified by the function or class name from "
                           "lasagne.nonlinearities or the custom "
                           "nonlinearities.py",
                      required=False, type=str, default=None)
    g1_1.add_argument('--nl_params',
                      help="parameters controling of the nonlinearity, "
                           "if applicable",
                      required=False, type=float, nargs='+', default=None)
    g1.add_argument('--disable_learn_init',
                    help="prevents the RNN from learning an initial RNN state",
                    required=False, action='store_true')
    
    # experiment_meta
    g2 = parser.add_argument_group("experiment_meta")
    g2.add_argument('--experiment_id',
                    help="name of experiment (and directory name)",
                    required=True, type=str)
    g2.add_argument('--experiment_dir',
                    help="path to directory containing experiment directory",
                    required=True, type=str)
    g2.add_argument('--data_rng_seed',
                    help="random number generator seed value for data",
                    required=False, type=int, default=None)
    g2.add_argument('--init_rng_seed',
                    required=False, type=int, default=None,
                    help="random number generator seed value for model init")
    g2.add_argument('--print_penalties',
                    required=False, action='store_true',
                    help="when training, report all optimization penalties "
                         "to stdout")
    g2.add_argument('--print_every_batch',
                    required=False, action='store_true',
                    help="when training, print an update after every batch")
    g2.add_argument('--track_gradient',
                    required=False, action='store_true',
                    help="when training, record all gradient values to file "
                         "(grad_track.h5)")
    g2.add_argument('--track_hidden_norms',
                    required=False, action='store_true',
                    help="when training, record all hidden norm magnitudes to "
                         "file (hidden_states_norm.h5)")
    g2.add_argument('--track_spectrum',
                    required=False, action='store_true',
                    help="when training a factorized_rnn, record all singular "
                         "values to file (spectrum.h5)")
    
    # dataset
    g3 = parser.add_argument_group("dataset")
    g3.add_argument('--dataset',
                    required=True, type=str,
                    choices=['add', 'copy', 'mnist', 'ptb'],
                    help="dataset on which to train and test")
    g3.add_argument('--data_dir',
                    required=False, type=str, default=".",
                    help="directory containing the dataset")
    g3.add_argument('--seq_length',
                    required=False, type=int, default=200,
                    help="for add or copy: the sequence length between the "
                         "input sequence and the output sequence\n"
                         "for ptb: the maximum input sequence length")
    g3.add_argument('--epoch_size',
                    required=False, type=int, default=100,
                    help="only add or copy: this number of batches forms one "
                         "epoch")
    g3.add_argument('--n_words_source',
                    required=False, type=int, default=10000,
                    help="only ptb: number of words to use for "
                         "training")
    g3.add_argument('--char_level',
                    required=False, action='store_true',
                    help="only ptb: character-level data instead of "
                         "word-level")
    g3_1 = g3.add_mutually_exclusive_group()
    g3_1.add_argument('--permute',
                      required=False, action='store_true',
                      help="only mnist: permute the order of input pixels "
                           "(permutation order is randomly generated once)")
    g3_1.add_argument('--always_permute',
                      required=False, action='store_true',
                      help="only mnist: permute the order of input pixels "
                           "(permutation order is randomly generated for each "
                           "input)")
    g3.add_argument('--scale',
                    required=False, type=int, default=None,
                    help="only mnist: scale image intensities by this amount")
    
    
    # optimization
    g4 = parser.add_argument_group('optimization')
    g4.add_argument('--batch_size',
                    required=True, type=int,
                    help="minibatch size for training, validation, testing")
    g4.add_argument('--num_epochs',
                    required=True, type=int,
                    help="number of epochs to train for")
    g4.add_argument('--clip_gradients',
                    required=False, type=float, default=100.,
                    help="clip gradient magnitudes to this value if they "
                         "exceed it")
    g4.add_argument('--orth_optimizer_type',
                    required=False, type=str, default='geoSGD',
                    choices=['geoSGD',],
                    help="optimizer to use for orthogonal matrices")
    g4.add_argument('--orth_learning_rate',
                    required=False, type=float, default=1e-6,
                    help="learning rate for orthogonal matrices")
    g4.add_argument('--free_optimizer_type',
                    required=False, type=str, default='RMSprop',
                    choices=['SGD', 'RMSprop', 'adam'],
                    help="optimizer to use for all weights other than those "
                         "in orthogonal matrices")
    g4.add_argument('--free_learning_rate',
                    required=False, type=float, default=1e-4,
                    help="learning rate for all weights other than those "
                         "in orthogonal matrices")
    g4.add_argument('--spectrum_learning_rate',
                    required=False, type=float, default=1e-4,
                    help="learning rate to use for the singular values of a "
                         "factorized_rnn - if not set, defaults to "
                         "free_learning_rate")
    g4.add_argument('--disable_rescale_spectral_updates',
                    required=False, action='store_true',
                    help="in factorized_rnn, spectral updates are rescaled to "
                         "to account for gradient scaling introduced by the "
                         "sigmoidal spectrum parameterization - this option "
                         "disables this rescaling, making spectral update "
                         "magnitude dependent on the margin size")
    g4.add_argument('--W_orth_penalty',
                    required=False, type=float, default=0.,
                    help="soft penalty weight for the devation of the "
                         "hidden-to-hidden transition matrix from "
                         "orthogonality (W'W = I)")
    g4.add_argument('--basis_orth_penalty',
                    required=False, type=float, default=0.,
                    help="soft penalty weight for the devation of basis "
                         "matrices in factorized_rnn from orthogonality "
                         "(M'M = I)")
    g4.add_argument('--spectral_orth_penalty',
                    required=False, type=float, default=0.,
                    help="soft penalty for the devation of singular values "
                         "in factorized_rnn from 1 (Gaussian prior)")
    g4.add_argument('--weight_decay',
                    required=False, type=float, default=0.0001,
                    help="soft penalty for the growth of any weight matrix "
                         "magnitude (inlcuding a composite transition "
                         "matrix formed from its factorized form)")
    
    return parser.parse_args()


"""
Get a nonlinearity function from a string name and an optional parameter value,
if applicable.
"""
def get_nonlinearity(nl_type, nl_params=None):
    if nl_type is None:
        return None
    
    func = None
    if func is None:
        try:
            # Try to get custom nonlinearity
            func = getattr(custom_nonlinearities, nl_type)
        except AttributeError:
            pass
    if func is None:
        try:
            # Try to get lasagne nonlinearity
            func = getattr(lasagne.nonlinearities, nl_type)
        except AttributeError:
            pass
    if func is None:
        raise ValueError("Specified nonlinearity ({}) not found."
                         "".format(nl_type))
    
    # Set parameters
    if nl_params is not None:
        func = func(*tuple(nl_params))
    
    return func


"""
Get an initialization method from a string name.
"""
def get_init(init_name):
    func = None
    if func is None:
        try:
            # Try to get custom nonlinearity
            func = getattr(init, init_name)
        except AttributeError:
            pass
    if func is None:
        try:
            # Try to get lasagne init
            func = getattr(lasagne.init, init_name)
        except AttributeError:
            pass
    if func is None:
        raise ValueError("Specified W_hid_to_hid init ({}) not found."
                         "".format(init_name))
    
    # Make sure that the output is an initializer
    if not issubclass(func, lasagne.init.Initializer):
        raise ValueError("Invalid init ({}). Expecting subclass of "
                         "\'lasagne.init.Initializer\', got {}"
                         "".format(init_name, func))

    return func


"""
Check if results directory exists and if so, interacts with the user to 
determine how to proceed.

Return True if training is to be resumed; False, otherwise.
"""
def handle_existing_directory(results_dir):
    resume = False
    
    if os.path.exists(results_dir):
        print("")
        print("WARNING! Results directory exists: \"{}\"".format(results_dir))
        write_into = None
        while write_into not in ['y', 'n', 'r', 'c', '']:
            write_into = str.lower(input( \
                "Write into existing directory?\n"
                "    y : yes\n"
                "    n : no (default)\n"
                "    r : delete and replace directory\n"
                "    c : continue/resume training\n"))
        if write_into in ['n', '']:
            print("Aborted")
            sys.exit()
        if write_into=='r':
            print("WARNING: Deleting existing results directory.")
            shutil.rmtree(results_dir)
        if write_into=='c':
            print("Attempting to load weights and continue training.")
            resume = True
        print("")
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    return resume


"""
Functions to save and load model weights to file.
"""
def save_model(model, weights_path):
    if os.path.exists(weights_path):
        # Copy old weights file just in case the overwrite fails
        shutil.copy(weights_path, weights_path+".prev")
    f = None
    try:
        f = h5py.File(weights_path, 'w')
        params = lasagne.layers.get_all_params(model['output'],
                                               unwrap_shared=True,
                                               trainable=True)
        def add_item(arr, item, key=None):
            if key is None:
                key = item.name
            arr.create_dataset(name=key,
                               shape=item.shape.eval(),
                               dtype=item.dtype,
                               data=item.eval(),
                               chunks=True,
                               compression='gzip')
        f.create_group('weights')
        for p in params:
            add_item(f['weights'], item=p, key=p.name)
        if 'updates' in model:
            f.create_group('optimizer_state')
            for idx, key in enumerate(model['updates']):
                if key not in params:
                    # Save only optimizer state, not update expressions.
                    add_item(f['optimizer_state'], item=key, key=str(idx))
    except:
        print("ERROR - Failed to save weights.")
        raise
    finally:
        if f is not None:
            f.close()
            
            
def load_model(model, weights_path):
    model_params = lasagne.layers.get_all_params(model['output'],
                                                 unwrap_shared=True,
                                                 trainable=True)
    loaded_params = {}
    f = None
    try:
        f = h5py.File(weights_path, 'r')
        for key in f['weights'].keys():
            loaded_params[key] = f['weights'][key][...]
        if 'optimizer_state' in f.keys() and 'updates' in model:
            updates = model['updates']
            updates_keys = list(updates.keys())
            for idx_str in f['optimizer_state'].keys():
                idx = int(idx_str)
                key = updates_keys[idx]
                key.set_value(np.array(f['optimizer_state'][idx_str]))
            model['updates'] = updates      # redundant line
    except:
        print("ERROR - Failed to load weights.")
        raise
    finally:
        if f is not None:
            f.close()
    for param in model_params:
        if param.name in loaded_params:
            param.set_value(loaded_params[param.name])
        else:
            print("WARNING: {} not in loaded parameters and "
                  "could not be set".format(param.name))
    
    return model
    
    
"""
Pickling functions.
"""
def pickle_save(obj, filepath, overwrite=True):
    if not overwrite and os.path.exists(filepath):
        raise IOError("ERROR: file exists, not overwriting {}"
                        "".format(filepath))
    f = None
    try:
        f = open(filepath, 'wb')
        pickle.dump(obj, f)
    except:
        print("ERROR: failed to pickle {}".format(filepath))
        raise
    finally:
        if f is not None:
            f.close()
    
    
def pickle_load(filepath):
    if not os.path.exists(filepath):
        raise IOError("ERROR: file does not exists, not loading {}"
                        "".format(filepath))
    f = None
    try:
        f = open(filepath, 'rb')
        obj = pickle.load(f)
    except:
        print("ERROR: failed to unpickle {}".format(filepath))
        raise
    finally:
        if f is not None:
            f.close()
    return obj


"""
Class for managing random number generators. Access, load, save.
"""
class rng_manager(object):
    def __init__(self):
        self.rng_dict = {}
        self.rng_path_dict = {}
    
    def add(self, name, rng=None, path=None):
        if name is None:
            raise ValueError("name must not be None")
        self.rng_dict[name] = rng
        self.rng_path_dict[name] = path
        
    def rem(self, name):
        self.rng_dict.pop(name)
        self.rng_path_dict.pop(name)
        
    def save(self, name=None):
        if name is None:
            # save all (NOTE: name is never None)
            for name in self.rng_dict:
                self.save(name)
        else:
            if self.rng_path_dict[name] is None:
                raise ValueError("save path not specified for rng \"{}\""
                                 "".format(name))
            pickle_save(self.rng_dict[name], self.rng_path_dict[name])
            
    def load(self, name=None):
        if name is None:
            # load all (NOTE: name is never None)
            for name in self.rng_dict:
                self.load(name)
        else:
            if self.rng_path_dict[name] is None:
                raise ValueError("load path not specified for rng \"{}\""
                                 "".format(name))
            self.rng_dict[name] = pickle_load(self.rng_path_dict[name])
            
    def __getitem__(self, idx):
        return self.rng_dict[idx]


"""
Define a numerically stable crossentropy function that takes tensor inputs.
"""
def ND_crossentropy(p, t, num_classes):
    _EPSILON = 1e-7
    if t.ndim == p.ndim:
        t_flat = t.reshape((-1, num_classes))
    else:
        t_flat = t.flatten()
    p_flat = p.reshape((-1, num_classes))
    p_flat = T.clip(p_flat, _EPSILON, 1.0 - _EPSILON)
    return theano.tensor.nnet.categorical_crossentropy(p_flat, t_flat)


"""
Dataset-specific model setup.
"""
def get_model_kwargs(model_params):
    model_kwargs = {'num_units': model_params['num_units'],
                    'W_in_to_hid': lasagne.init.Uniform(),
                    'b': lasagne.init.Constant(model_params['bias']),
                    'grad_clipping': model_params['grad_clipping'],
                    'learn_init': model_params['learn_init'],
                    'nonlinearity': model_params['nonlinearity']}
    if model_params['model_type']=='factorized_rnn':
        model_kwargs['hard_spectral_boundary'] = \
            model_params['hard_spectral_boundary']
    else:
        model_kwargs['W_hid_to_hid'] = model_params['W_hid_to_hid']
    return model_kwargs
        

def setup_add(model_params, rng):
    '''
    Addition task.
    '''
    
    # Prepare model parameters
    model_kwargs = get_model_kwargs(model_params)
    
    # Data generator
    data_gen = gen_data_add(seq_length=global_params['seq_length'],
                            batch_size=global_params['batch_size'],
                            epoch_len=global_params['epoch_size'],
                            rng=rng['data'])
    
    # Input    
    input_shape = (global_params['batch_size'], global_params['seq_length'], 2)
    input = lasagne.layers.InputLayer(shape=input_shape)
    zeros_input_shape = input_shape[:-1]+(model_params['num_units'],)
    zeros_input = lasagne.layers.InputLayer(shape=zeros_input_shape,
                                 input_var=T.zeros(shape=zeros_input_shape,
                                                   dtype=theano.config.floatX))
    model_kwargs['incoming'] = input
    
    # RNN
    model_kwargs['only_return_final'] = True
    model_kwargs['gt_zeros'] = zeros_input
    rnn = model_types[model_params['model_type']](**model_kwargs)
    
    # Output
    output_dense = lasagne.layers.DenseLayer(rnn, num_units=1,
                                             nonlinearity=None)
    output = output_dense
    
    # Cost and metrics
    metrics = OrderedDict()
    predicted_values = lasagne.layers.get_output(output)
    target_values = T.vector('target_output')
    cost_vec = (predicted_values - target_values)**2
    cost = T.mean(cost_vec)
    metrics['cost'] = cost
     
    # Gradient with respect to the last element in the output.
    grad_hiddens = T.grad(cost_vec[-1], zeros_input.input_var)
    
    # Set up model
    model = {'input': input,
             'input_mask': None,
             'output': output,
             'target': target_values,
             #'grad_hiddens': grad_hiddens,
             'rnn': rnn,
             'hid_to_out': output_dense.W,
             'cost': cost,
             'metrics': metrics,
             'data_gen': data_gen,
             'track_key': None,
             'track_func': None}
    
    return model
    
    
def setup_copy(model_params, rng):
    '''
    Copy task.
    
    Note: For an explanation of the reshape madness below, see
    http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    
    It is done to be able to apply a dense layer and softmax to the output of
    each time step.
    '''
    
    # Prepare model parameters
    model_kwargs = get_model_kwargs(model_params)
    
    # Data generator
    data_gen = gen_data_copy(seq_length=global_params['seq_length'],
                             batch_size=global_params['batch_size'],
                             epoch_len=global_params['epoch_size'],
                             rng=rng['data'])
    
    # Input
    input_shape = (global_params['batch_size'],
                   global_params['seq_length']+20, 1)
    input = lasagne.layers.InputLayer(shape=input_shape)
    zeros_input_shape = input_shape[:-1]+(model_params['num_units'],)
    zeros_input = lasagne.layers.InputLayer(shape=zeros_input_shape,
                                 input_var=T.zeros(shape=zeros_input_shape,
                                                   dtype=theano.config.floatX))
    model_kwargs['incoming'] = input
    
    # RNN
    model_kwargs['only_return_final'] = False
    model_kwargs['gt_zeros'] = zeros_input
    rnn = model_types[model_params['model_type']](**model_kwargs)
    rnn_out = lasagne.layers.ReshapeLayer(rnn, (-1, [2]))
    
    # Output
    output_dense = lasagne.layers.DenseLayer(rnn_out, num_units=9,
                                   nonlinearity=lasagne.nonlinearities.softmax)
    output = lasagne.layers.ReshapeLayer(output_dense,
              (global_params['batch_size'], global_params['seq_length']+20, 9))
    
    # Cost and metrics
    metrics = OrderedDict()
    predicted_values = lasagne.layers.get_output(output)
    target_values = T.itensor3('target_output')
    cost_vec = ND_crossentropy(predicted_values, target_values,
                               num_classes=9)
    cost = T.mean(cost_vec)
    metrics['cost'] = cost
    acc = T.mean(T.eq(T.argmax(predicted_values, axis=-1)[:,-10:],
                      T.argmax(target_values, axis=-1)[:,-10:]))
    metrics['accuracy'] = acc
    
    # Gradient with respect to the last element in the output.
    grad_hiddens = T.grad(cost_vec[-1], zeros_input.input_var)
    
    # Set up model
    model = {'input': input,
             'input_mask': None,
             'output': output,
             'target': target_values,
             'grad_hiddens': grad_hiddens,
             'rnn': rnn,
             'hid_to_out': output_dense.W,
             'cost': cost,
             'metrics': metrics,
             'data_gen': data_gen,
             'track_key': None,
             'track_func': None}
    
    return model
    
    
def setup_mnist(model_params, rng, data_dir, results_dir):
    '''
    Sequential MNIST
    '''
    
    # Prepare model parameters
    model_kwargs = get_model_kwargs(model_params)
    
    # Prepare data
    f = gzip.open(os.path.join(data_dir, "mnist.pkl.gz"), 'rb')
    if sys.version_info >= (3,):
        # Python 3
        train_set, valid_set, test_set = pickle.load(f, encoding='iso-8859-1')
    else:
        # Python 2
        train_set, valid_set, test_set = pickle.load(f)
    new_shape = None
    if global_params['scale_mnist'] is not None:
        new_shape = np.array([28, 28])*global_params['scale_mnist']
        new_shape = np.round(new_shape).astype(np.uint8)
    vec_len = 28**2
    if new_shape is not None:
        vec_len = np.prod(new_shape)
        
    # Determine data permutation
    constant_permutation = None
    permutation_filepath = os.path.join(results_dir, "permutation.pkl")
    if os.path.exists(permutation_filepath):
        f = None
        try:
            f = open(permutation_filepath, 'rb')
            constant_permutation = pickle.load(f)
        except:
            print("ERROR: Failed to load mnist permutation.")
            raise
        finally:
            if f is not None:
                f.close()
    else:
        constant_permutation = rng['data'].permutation(vec_len)
        f = None
        try:
            f = open(permutation_filepath, 'wb')
            pickle.dump(constant_permutation, f)
        except:
            raise
        finally:
            if f is not None:
                f.close()
        
    # Data generator
    data_gen_kwargs = {'batch_size': global_params['batch_size'],
                       'permute': 0,
                       'new_shape': new_shape}
    if global_params['permute']==True:
        # randomly permute the data once
        data_gen_kwargs['permute'] = 1
    if global_params['always_permute']==True:
        # randomly permute every sample with a new permutation
        data_gen_kwargs['permute'] = 2
    data_gen_train = mnist_iterator(train_set,
                                    rng=copy.copy(rng['data']),
                                    shuffle=True,
                                    **data_gen_kwargs)
    data_gen_valid = mnist_iterator(valid_set,
                                    rng=copy.copy(rng['data']),
                                    shuffle=False,
                                    **data_gen_kwargs)
    data_gen_test = mnist_iterator(test_set,
                                   rng=copy.copy(rng['data']),
                                   shuffle=False,
                                   **data_gen_kwargs)
    data_gen = (data_gen_train, data_gen_valid, data_gen_test)
    
    # Input
    input_shape = (None, vec_len, 1)
    input = lasagne.layers.InputLayer(shape=input_shape)
    zeros_input_shape = input_shape[:-1]+(model_params['num_units'],)
    zeros_inferred_shape = (input.input_var.shape[0],
                            vec_len,
                            model_params['num_units'])
    zeros_input = lasagne.layers.InputLayer(shape=zeros_input_shape,
                                 input_var=T.zeros(shape=zeros_inferred_shape,
                                                   dtype=theano.config.floatX))
    model_kwargs['incoming'] = input
    
    # RNN
    model_kwargs['only_return_final'] = True
    model_kwargs['gt_zeros'] = zeros_input
    rnn = model_types[model_params['model_type']](**model_kwargs)
    
    # Output
    output_dense = lasagne.layers.DenseLayer(rnn, num_units=10,
                                   nonlinearity=lasagne.nonlinearities.softmax)
    output = output_dense
    
    # Cost and metrics
    metrics = OrderedDict()
    predicted_values = lasagne.layers.get_output(output)
    target_values = T.ivector('target_output')
    cost_vec = ND_crossentropy(predicted_values, target_values,
                               num_classes=10)
    cost = T.mean(cost_vec)
    metrics['cost'] = cost
    acc = T.mean(T.eq(T.argmax(predicted_values, axis=-1), target_values))
    metrics['accuracy'] = acc
                         
    # Gradient with respect to the last element in the output.
    grad_hiddens = T.grad(cost_vec[-1], zeros_input.input_var)
    
    # Set up model
    model = {'input': input,
             'input_mask': None,
             'output': output,
             'target': target_values,
             'grad_hiddens': grad_hiddens,
             'rnn': rnn,
             'hid_to_out': output_dense.W,
             'cost': cost,
             'metrics': metrics,
             'data_gen': data_gen,
             'track_key': 'accuracy_v',
             'track_func': max}
    
    return model
    
    
def setup_ptb(model_params, rng, data_dir):
    '''
    Penn Treebank word/character task.
    '''
    
    # Prepare model parameters
    model_kwargs = get_model_kwargs(model_params)
    
    # Data generator
    time_steps = global_params['seq_length']
    file_paths = {'dict':  os.path.join(data_dir, "dict.pkl"),
                  'train': os.path.join(data_dir, "ptb.train.txt"),
                  'valid': os.path.join(data_dir, "ptb.valid.txt"),
                  'test':  os.path.join(data_dir, "ptb.test.txt")}
    ptb_iterator_kwargs = {'source_dict': file_paths['dict'],
                           'batch_size': global_params['batch_size'],
                           'maxlen': time_steps,
                           'n_words_source': global_params['n_words_source']}
    train_vocab_len = global_params['n_words_source']
    if global_params['char_level']==True:
        ptb_iterator_kwargs['char_level'] = True
        train_vocab_len = 49
    data_gen_train = ptb_iterator(file_paths['train'],
                                  rng=copy.copy(rng['data']),
                                  shuffle=True,
                                  **ptb_iterator_kwargs)
    data_gen_valid = ptb_iterator(file_paths['valid'],
                                  rng=copy.copy(rng['data']),
                                  shuffle=False,
                                  **ptb_iterator_kwargs)
    data_gen_test  = ptb_iterator(file_paths['test'],
                                  rng=copy.copy(rng['data']),
                                  shuffle=False,
                                  **ptb_iterator_kwargs)         
    data_gen = (data_gen_train, data_gen_valid, data_gen_test)
    
    # Input
    input_shape = (None, time_steps)
    input = lasagne.layers.InputLayer(shape=input_shape,
                                      input_var=T.imatrix())
    input_mask = lasagne.layers.InputLayer(shape=input_shape)
    embed = lasagne.layers.EmbeddingLayer(input,
                                         input_size=train_vocab_len,
                                         output_size=model_params['num_units'])
    embed.W.name = 'W_embed'
    zeros_input_shape = input_shape+(model_params['num_units'],)
    zeros_inferred_shape = (input.input_var.shape[0],
                            time_steps,
                            model_params['num_units'])
    zeros_input = lasagne.layers.InputLayer(shape=zeros_input_shape,
                                 input_var=T.zeros(shape=zeros_inferred_shape,
                                                   dtype=theano.config.floatX))
    model_kwargs['incoming'] = embed
    model_kwargs['mask_input'] = input_mask
    
    # RNN
    model_kwargs['only_return_final'] = False
    model_kwargs['gt_zeros'] = zeros_input
    rnn = model_types[model_params['model_type']](**model_kwargs)
    rnn_out = lasagne.layers.ReshapeLayer(rnn, (-1, [2]))
    
    # Output
    output_dense = lasagne.layers.DenseLayer(rnn_out,
                                   num_units=train_vocab_len,
                                   nonlinearity=lasagne.nonlinearities.softmax)
    output = lasagne.layers.ReshapeLayer(output_dense,
                                             (-1, time_steps, train_vocab_len))
    
    # Cost and metrics
    metrics = OrderedDict()
    predicted_values = lasagne.layers.get_output(output)
    target_values = T.matrix('target_output', dtype='int32')
    masked_idx = input_mask.input_var.nonzero()

    cost_vec = ND_crossentropy(predicted_values[masked_idx],
                               target_values[masked_idx],
                               num_classes=train_vocab_len)
    cost_vec /= T.log(2)    # adjust natural log to base 2
    cost = T.mean(cost_vec)
    metrics['cost'] = cost
    acc = T.mean(T.eq(T.argmax(predicted_values[masked_idx], axis=-1),
                      target_values[masked_idx]))
    metrics['accuracy'] = acc
    metrics['perplexity'] = 2**cost
    
    # Gradient with respect to the last element in the output.
    grad_hiddens = T.grad(cost_vec[-1], zeros_input.input_var)
    
    # Set up model
    model = {'input': input,
             'input_mask': input_mask,
             'output': output,
             'target': target_values,
             'grad_hiddens': grad_hiddens,
             'rnn': rnn,
             'hid_to_out': output_dense.W,
             'cost': cost,
             'metrics': metrics,
             'data_gen': data_gen,
             'track_key': 'accuracy_v',
             'track_func': max}
    
    return model


"""
Function to add penalties to a cost function, given the model.
"""
def add_penalties(model):
    '''
    Modify model cost by adding penalty terms:
    
    - hidden-to-hidden transition non-orthogonality penalty
    - basis non-orthogonality penalty
    - spectral non-orthogonality penalty
    - L2 weight norm -- weight decay
    '''
    
    # Convenience functions
    def get_params(**flag_kwargs):
        return lasagne.layers.get_all_params(model['output'],
                                             unwrap_shared=False,
                                             regularizable=True,
                                             **flag_kwargs)
    penalty = lasagne.regularization.apply_penalty
    
    # Penalties
    W_orth_penalty = penalties.orthogonality(model['rnn'].W_hid_to_hid)
    basis_orth_penalty = penalty(get_params(basis=True),
                                 penalties.orthogonality)
    spectral_orth_penalty = penalty(get_params(spectrum=True),
                                    penalties.constant_l2(1.))
    l2_params = [model['rnn'].W_in_to_hid,
                 model['rnn'].W_hid_to_hid,
                 model['hid_to_out']]
    l2_penalty = penalty(l2_params, penalties.constant_l2(0))
    
    
    
    # Update cost.
    model['cost'] = model['cost'] \
        + W_orth_penalty*global_params['W_orth_penalty'] \
        + basis_orth_penalty*global_params['basis_orth_penalty'] \
        + spectral_orth_penalty*global_params['spectral_orth_penalty'] \
        + l2_penalty*global_params['weight_decay']
     
    return model


"""
Set up update rules.
"""
def setup_updates(model, margin=None):
    all_trainable_params = lasagne.layers.get_all_params(model['output'],
                                                         unwrap_shared=True,
                                                         trainable=True)
    updates = OrderedDict()

    # These parameters will be updated with the orthogonal updater.
    orthogonal_params = []
    if model_params['model_type']=='simple_rnn':
        orthogonal_params = [model['rnn'].W_hid_to_hid]
    elif model_params['model_type']=='factorized_rnn':
        orthogonal_params = [model['rnn'].hidden_to_hidden.W_u,
                             model['rnn'].hidden_to_hidden.W_v]
        
    if global_params['orth_optimizer_type']=='geoSGD':
        updates_orth = geoSGD(loss_or_grads=model['cost'],
                             params=orthogonal_params,
                             learning_rate=global_params['orth_learning_rate'])
        updates.update(updates_orth)
    elif global_params['orth_optimizer_type'] is None:
        orthogonal_params = []
    else:
        raise ValueError("Optimizer type {} unrecognized."
                        "".format(global_params['orth_optimizer_type']))
        
    # These parameters have no orthogonality constraints and can be updated
    # by a different updater.
    nonorthogonal_params = [p for p in all_trainable_params \
                            if p not in orthogonal_params]


    # Should the spectral parameter updates be rescaled?
    # 
    # This is needed because given the parameterization of the spectral values,
    # the update steps are scaled by the margin. Here they are rescaled back.
    separate_spectrum = False
    if model_params['model_type']=='factorized_rnn' and margin is not None:
        separate_spectrum = True

    # Separate out the spectrum parameters
    if separate_spectrum:
        spectrum = model['rnn'].hidden_to_hidden.W_s_params
        nonorthogonal_params = [p for p in nonorthogonal_params if p!=spectrum]
        
    # Set spectrum learning rate
    if global_params['spectrum_learning_rate'] is None:
        global_params['spectrum_learning_rate'] = \
                                                global_params['free_learning_rate']
    elif margin is not None:
        if global_params['rescale_spectral_updates'] and margin>0:
            global_params['spectrum_learning_rate'] = \
                                global_params['spectrum_learning_rate']/(2.*margin)
                        

    # Create update dictionary
    if global_params['free_optimizer_type']=='SGD':
        updates_free = lasagne.updates.nesterov_momentum( \
                                loss_or_grads=model['cost'],
                                params=nonorthogonal_params,
                                learning_rate=global_params['free_learning_rate'],
                                momentum=0.9)
        updates.update(updates_free)
        if separate_spectrum:
            updates_free_spectrum = lasagne.updates.nesterov_momentum( \
            loss_or_grads=model['cost'],
            params=[model['rnn'].hidden_to_hidden.W_s_params],
            learning_rate=global_params['spectrum_learning_rate'],
            momentum=0.9)
            updates.update(updates_free_spectrum)
    elif global_params['free_optimizer_type']=='RMSprop':
        updates_free = lasagne.updates.rmsprop( \
                                loss_or_grads=model['cost'],
                                params=nonorthogonal_params,
                                learning_rate=global_params['free_learning_rate'],
                                rho=0.9)
        updates.update(updates_free)
        if separate_spectrum:
            updates_free_spectrum = lasagne.updates.rmsprop( \
            loss_or_grads=model['cost'],
            params=[model['rnn'].hidden_to_hidden.W_s_params],
            learning_rate=global_params['spectrum_learning_rate'],
            rho=0.9)
            updates.update(updates_free_spectrum)
    elif global_params['free_optimizer_type']=='adam':
        updates_free = lasagne.updates.adam( \
                                loss_or_grads=model['cost'],
                                params=nonorthogonal_params,
                                learning_rate=global_params['free_learning_rate'],
                                beta1=0.9, beta2=0.999)
        updates.update(updates_free)
        if separate_spectrum:
            updates_free_spectrum = lasagne.updates.adam( \
            loss_or_grads=model['cost'],
            params=[model['rnn'].hidden_to_hidden.W_s_params],
            learning_rate=global_params['spectrum_learning_rate'],
            beta1=0.9, beta2=0.999)
            updates.update(updates_free_spectrum)
    else:
        raise ValueError("Optimizer type {} unrecognized."
                        "".format(global_params['free_optimizer_type']))
    
    return updates


"""
Training code.
"""
def train(model, rng, results_dir, resume=False):
    
    # Setup history
    history_valid = {}
    history = {'cost': []}
    for key in model['metrics']:
        history[key] = []
        history_valid[key+'_v'] = []
    log_file_path = os.path.join(results_dir, "training_log.txt")
    
    if resume:        
        # Rebuild history from log file
        print("Rebuilding history from log file.")
        def scrub(path, keys):
            history = OrderedDict()
            for key in keys:
                history[key] = []
                
            with open(path, 'rt') as f:
                for l in f:
                    for key in keys:
                        if key in l:
                            history[key].append( float(l.split('=')[-1]) )
            return history
        history = scrub(log_file_path, keys=model['metrics'].keys())
        history_valid = scrub(log_file_path,
                              keys=[key+'_v' for key in model['metrics']])
        
    # Determine start epoch
    start_epoch = 0
    if resume:
        # Get the epoch number
        print("Getting last epoch number from log file.")
        log_file = None
        try:
            log_file = open(os.path.join(results_dir, "training_log.txt"),
                            'rt')
            for line in log_file:
                if 'Epoch' in line:
                    start_epoch += 1
        except:
            print("Failed to read training log.")
            raise
        finally:
            if log_file is not None:
                log_file.close()
                
    
    # Setup up value tracking (gradients, norms).
    track_vals = OrderedDict()
    track_file = None
    def make_h5_ds(name, element_shape, track_file):
        if track_file is None:
            track_file = h5py.File(os.path.join(results_dir, "tracker.h5"),
                                   'a')
        if name not in track_file:
            track_file.create_dataset(name, (0,)+tuple(element_shape),
                                      maxshape=(None,)+tuple(element_shape),
                                      compression='gzip', compression_opts=9)
        return track_file
    if (global_params['track_gradient'] or \
        global_params['track_hidden_norms'] or \
        global_params['track_spectrum']) \
    and os.path.exists(os.path.join(results_dir, "tracker.h5")):
        track_file = h5py.File(os.path.join(results_dir, "tracker.h5"), 'a')
    if global_params['track_gradient']:
        track_vals['grad_track'] = T.sum(model['grad_hiddens']**2, axis=(0,2))
        slen = model['input'].shape[1]
        track_file = make_h5_ds('grad_track', [slen], track_file)
    if global_params['track_hidden_norms']:
        # Here we are going to save the hidden state tensor norm
        # and also the norm of the cost function gradient with 
        # respect to the hidden states.
        # Note : Lasagne uses the following ordering to represent
        # time series data (n_batch, n_time_steps, n_features).
        # In order to sum the norm for each time step, we need to
        # sum over the axis representing the batch instances and
        # the features.
        hidden_states = lasagne.layers.get_output(model['rnn'])
        track_vals['hidden_states_norm'] = (hidden_states ** 2).sum(axis=(0,2))
        slen = model['input'].shape[1]
        track_file = make_h5_ds('hidden_states_norm', [slen], track_file)
    if global_params['track_spectrum'] and \
    model_params['model_type']=='factorized_rnn':
        spectral_params = lasagne.layers.get_all_params(model['output'],
                                                        unwrap_shared=False,
                                                        spectrum=True)
        assert(len(spectral_params)==1)
        track_vals['spectrum'] = spectral_params[0]
        track_file = make_h5_ds('spectrum', [model_params['num_units']], 
                                track_file)
        
        
    # Setup cost and training functions.
    if model['input_mask'] is not None:
        input_list = [model['input'].input_var,
                      model['input_mask'].input_var,
                      model['target']]
    else:
        input_list = [model['input'].input_var, model['target']]
    metric_list = list(model['metrics'][key] for key in model['metrics'])
    if len(track_vals):
        metric_list += list(track_vals[key] for key in track_vals)
    predicted_values = lasagne.layers.get_output(model['output'])
    predict = theano.function(input_list[:-1],
                              T.argmax(predicted_values, axis=-1))
    train = theano.function(input_list, metric_list, updates=model['updates'])
    
    
    # Helper function to evaluate metrics.
    def evaluate(key_suffix, data_gen, metrics, input_list):
        metric_list = list(metrics[key] for key in metrics)
        compute_metrics = theano.function(input_list, metric_list)
        metrics_eval = OrderedDict()
        for key in metrics:
            metrics_eval[key+key_suffix] = 0
        num_samples = 0
        for batch in data_gen():
            num_samples += len(batch[-1])
            metrics_eval_list = compute_metrics(*batch)
            for i, key in enumerate(metrics_eval):
                metrics_eval[key] += metrics_eval_list[i]*len(batch[-1])
        for key in metrics_eval:
            metrics_eval[key] /= float(num_samples)
        return metrics_eval
    
    
    # Open log file for writing.
    log_file = open(os.path.join(results_dir, "training_log.txt"), 'at')
    
    
    # Prepare data iterators.
    data_gen_train = None
    data_gen_valid = None
    data_gen_test  = None
    if not hasattr(model['data_gen'], '__len__'):
        data_gen_train = model['data_gen']
    else:
        data_gen_train, data_gen_valid, data_gen_test = model['data_gen']
    assert data_gen_train is not None
    
    
    # Set up training loop.
    def train_epoch(epoch):
        # Set up mettric accumulator
        accum_vals = OrderedDict(( ('cost', 0), ))
        for key in model['metrics']:
            accum_vals[key] = 0
        num_train_samples = 0
        
        # Train on each batch, accumulating metrics
        for batch_num, batch in enumerate(data_gen_train()):
            num_train_samples += len(batch[-1])
            
            # The train() function also computes all metrics
            batch_metrics = OrderedDict()
            batch_metrics_list = train(*batch)
            
            # Record and accumulate all metrics (and stop if NaN in cost)
            for i, key in enumerate(model['metrics']):
                batch_metrics[key] = batch_metrics_list[i]
            if np.isnan(batch_metrics['cost']):
                raise RuntimeError("NaN !!")
            print_msg = "epoch {} batch {}: ".format(epoch, batch_num)
            for key in batch_metrics:
                print_msg += " {} = {:.3g} ".format(str(key),
                                                    float(batch_metrics[key]))
                accum_vals[key] += batch_metrics[key]*len(batch[-1])
                
            # If enabled, print the batch metrics
            if global_params['print_every_batch']:
                print(print_msg)
                print(print_msg, file=log_file)
                
            # Record gradients and norms, if enabled
            for i, key in enumerate(track_vals):
                j = len(model['metrics'])
                idx_start = len(track_file[key])
                idx_end = idx_start + len(batch)
                track_file[key].resize(idx_end, axis=0)
                track_file[key][idx_start:idx_end] = batch_metrics_list[j+i]
        
        # Save weights
        save_model(model,
                   weights_path=os.path.join(results_dir, "weights.h5"))
        
        # Save RNG
        rng.save('data')
            
        # Print+log metrics and record their history
        for key in accum_vals:
            accum_vals[key] /= num_train_samples
            history[key].append(accum_vals[key])
        print_msg = "Epoch {} :".format(epoch)
        for key in model['metrics'].keys():
            print_msg += "\n    {} = {}".format(str(key), accum_vals[key])
            
        # Add validation metrics if there is validation data
        if data_gen_valid is not None:
            metrics_valid = evaluate(key_suffix='_v',
                                     data_gen=data_gen_valid,
                                     metrics=model['metrics'],
                                     input_list=input_list)
            print_msg += "\nVALIDATION:"
            for key in metrics_valid.keys():
                val = metrics_valid[key]
                print_msg += "\n    {} = {}".format(str(key), val)
                if key not in history_valid.keys():
                    history_valid[key] = []
                history_valid[key].append(val)
            
            # If there is an accuracy metric, save model with highest
            # validation accuracy.
            track_key, track_func = model['track_key'], model['track_func']
            if track_key in metrics_valid.keys():
                if history_valid[track_key][-1] == \
                                          track_func(history_valid[track_key]):
                    save_model(model,
                               weights_path=os.path.join(results_dir,
                                                         "best_weights.h5"))
            
        # Add other metrics
        if global_params['print_penalties']:
            def val(x):
                tensortype = theano.tensor.TensorVariable
                return x.eval() if isinstance(x, tensortype) else x
            print_penalties_msg = "PENALTIES:" \
                +"\n{}={}\n{}={}\n{}={}\n{}={}".format(\
                "W_orth_penalty", val(W_orth_penalty),
                "basis_orth_penalty", val(basis_orth_penalty),
                "spectral_orth_penalty", val(spectral_orth_penalty),
                "weight_decay", val(l2_penalty))
            print_msg += "\n"+print_penalties_msg
            print_msg += "\n{}={}".format("norm_h2h", norm_hid_to_hid.eval())
            if len(spectral_params)>0:
                spectrum = spectral_params[0].eval()
                print_msg += "\n{}={}".format("max_sv", np.max(spectrum))
                print_msg += "\n{}={}".format("min_sv", np.min(spectrum))
                print_msg += "\n{}={}".format("mean_sv", np.mean(spectrum))
                print_msg += "\n{}={}".format("std_sv", np.std(spectrum))
            print_msg += "\n"
            
        print(print_msg)
        print(print_msg, file=log_file)
        
    # Train
    try:
        for epoch in range(start_epoch, global_params['num_epochs']):
            train_epoch(epoch)
    
        # If there is a test set, evaluate all metrics on it
        if data_gen_test is not None:
            print("\nCOMPUTING TEST SET METRICS")
            print("\nCOMPUTING TEST SET METRICS", file=log_file)
            
            def run_test(print_msg):
                metrics_test = evaluate(key_suffix='_test',
                                        data_gen=data_gen_test,
                                        metrics=model['metrics'],
                                        input_list=input_list)
                for key in metrics_test.keys():
                    val = metrics_test[key]
                    print_msg += "\n    {} = {}".format(str(key), val)
                return print_msg
            
            print_msg = run_test("Test results on final weights:")
            model = load_model(model,
                               weights_path=os.path.join(results_dir,
                                                         "best_weights.h5"))
            print_msg += "\n"
            print_msg += run_test("Test results on best weights:")
            print(print_msg)
            print(print_msg, file=log_file)
        
    except KeyboardInterrupt:
        pass
    finally:
        if log_file is not None and not log_file.closed:
            log_file.close()
            log_file = None
        if track_file is not None:
            track_file.close()
            track_file = None
            
    return history, history_valid, epoch



if __name__=='__main__':
    
    """
    Set up arguments
    """
    args = parse()
    
    model_types = {'simple_rnn' : RecurrentLayer,
                   'factorized_rnn' : FactorizedRecurrentLayer}
    
    model_params = OrderedDict((
        ('model_type', args.model_type),
        ('num_units', args.num_units),
        ('bias', args.bias),
        ('nonlinearity', get_nonlinearity(args.nonlinearity, args.nl_params)),
        ('W_hid_to_hid', get_init(args.W_hid_to_hid)),
        ('grad_clipping', args.clip_gradients),
        ('learn_init', not args.disable_learn_init),
        ('hard_spectral_boundary', args.margin),
        ))
        
    global_params = OrderedDict((
        # Experiment name (also results directory name)
        ('experiment_ID', args.experiment_id),
        
        # Input data
        ('dataset', args.dataset),
        ('n_words_source', args.n_words_source),
        ('scale_mnist', args.scale),
        ('seq_length', args.seq_length),
        ('char_level', args.char_level),
        ('permute', args.permute),
        ('always_permute', args.always_permute),
        
        # RNG
        ('data_rng_seed', args.data_rng_seed),
        ('init_rng_seed', args.init_rng_seed),
        
        # Penalties
        ('W_orth_penalty', args.W_orth_penalty),
        ('basis_orth_penalty', args.basis_orth_penalty),
        ('spectral_orth_penalty', args.spectral_orth_penalty),
        ('weight_decay', args.weight_decay),
        ('print_penalties', args.print_penalties),
        
        # Optimization settings
        ('batch_size', args.batch_size),
        ('orth_learning_rate', args.orth_learning_rate),
        ('orth_optimizer_type', args.orth_optimizer_type),
        ('free_learning_rate', args.free_learning_rate),
        ('free_optimizer_type', args.free_optimizer_type),
        ('spectrum_learning_rate', args.spectrum_learning_rate),
        ('epoch_size', args.epoch_size),
        ('num_epochs', args.num_epochs),
        ('rescale_spectral_updates',not args.disable_rescale_spectral_updates),
        
        # Other
        ('print_every_batch', args.print_every_batch),
        ('track_gradient', args.track_gradient),
        ('track_hidden_norms', args.track_hidden_norms),
        ('track_spectrum', args.track_spectrum)
        ))
    
    
    """
    Print settings to screen.
    """
    settings_str =  "Experiment: {}\n\n".format(global_params['experiment_ID'])
    for key in model_params.keys():
        settings_str += "{} : {}\n".format(key, model_params[key])
    settings_str += "\n"
    for key in global_params.keys():
        settings_str += "{} : {}\n".format(key, global_params[key])
    settings_str += "\n"
    print(settings_str)
    
    
    """
    Check if results directory exists. If it exists, ask the user what to do.
    Otherwise, create it and run training.
    """
    results_dir = os.path.join(args.experiment_dir, args.experiment_id)
    resume = handle_existing_directory(results_dir)
    
    
    """
    Write settings (previously outputted to screen) to file.
    """
    settings_file = open(os.path.join(results_dir, "settings.txt"), 'wt')
    print(settings_str, file=settings_file)
        
    
    """
    Set up random number generators.
    """
    rng = rng_manager()
    data_rng_filepath = os.path.join(results_dir, "data_rng.pkl")
    rng.add(name='data',
            rng=np.random.RandomState(global_params['data_rng_seed']),
            path=data_rng_filepath)
    if os.path.exists(data_rng_filepath):
        print("Reloading saved data RNG.")
        rng.load('data')
    init_rng_filepath = os.path.join(results_dir, "init_rng.pkl")
    rng.add(name='init',
            rng=np.random.RandomState(global_params['init_rng_seed']),
            path=init_rng_filepath)
    if os.path.exists(init_rng_filepath):
        print("Reloading saved init RNG.")
        rng.load('init')
    lasagne.random.set_rng(rng['init'])
    rng.save()

    
    """
    Set up model.
    """
    if args.dataset=='add':
        model = setup_add(model_params, rng)
    if args.dataset=='copy':
        model = setup_copy(model_params, rng)
    if args.dataset=='mnist':
        model = setup_mnist(model_params, rng,
                            data_dir=args.data_dir, results_dir=results_dir)
    if args.dataset=='ptb':
        model = setup_ptb(model_params, rng,
                          data_dir=args.data_dir)
    
        
    """
    Add penalties to cost.
    """
    model = add_penalties(model)
    
    
    """
    Set up update rules.
    """
    updates = setup_updates(model,
                            margin=model_params['hard_spectral_boundary'])
    model['updates'] = updates
    
    
    """
    If resuming, load model (including update rules and optimizer state).
    """
    if resume:
        model = load_model(model, os.path.join(results_dir, "weights.h5"))

    
    """
    Train.
    """
    history, history_valid, epochs = train(model, 
                                           rng=rng,
                                           results_dir=results_dir, 
                                           resume=resume)
    
    
    """
    Plot history.
    """
    fig = plt.figure()
    colors = ['red', 'blue', 'green', 'brown']
    for key, c in zip(model['metrics'].keys(), colors):
        plt.plot(history[key], color=c, label=key)
        if len(history_valid):
            plt.plot(history_valid[key+'_v'],
                     color=c, linestyle='--', label=key+'_v')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis([0, epochs, 0, 4])
    plt.xlabel('number of epochs')
    plt.title(global_params['experiment_ID'])
    plt.savefig(os.path.join(results_dir,
                             "history_"+global_params['experiment_ID']+".png"),
                bbox_inches='tight')
    
