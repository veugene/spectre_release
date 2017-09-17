# Spectral deviation from orthogonality

This repo contains code to implement and train a factorized RNN, as described in the paper:

*"On orthogonality and learning recurrent networks with long term dependencies"*

https://arxiv.org/abs/1702.00071

Eugene Vorontsov, Chiheb Trabelsi, Christopher Pal, Samuel Kadoury

## Abstract
It is well known that it is challenging to train deep neural networks and recurrent neural networks for tasks that exhibit long term dependencies. The vanishing or exploding gradient problem is a well known issue associated with these challenges. One approach to addressing vanishing and exploding gradients is to use either soft or hard constraints on weight matrices so as to encourage or enforce orthogonality. Orthogonal matrices preserve gradient norm during backpropagation and may therefore be a desirable property. This paper explores issues with optimization convergence, speed and gradient stability when encouraging or enforcing orthogonality. To perform this analysis, we propose a weight matrix factorization and parameterization strategy through which we can bound matrix norms and therein control the degree of expansivity induced during backpropagation. We find that hard constraints on orthogonality can negatively affect the speed of convergence and model performance.

## Factorized RNN
The transition matrix is factorized by construction in an SVD-style factorization, formed from two orthogonal basis matrices and a vector of singular values. The singular values can be hard bounded, limiting the composite transition matrix representations to within some small distance about the sub-manifold of orthogonal matrices (Stiefel manifold). This parameterization is intended to allow an analysis on deviation from orthogonality.

We find that deviating from this sub-manifold can allow faster convergence and better model performance while still retaining approximate norm preservation, a benefit guaranteed by pure orthogonality.

Details are in the paper.

## Datasets in this code

**Add** : auto-generated.

**Copy** : auto-generated.

**Sequential MNIST** : dataset can be downloaded from link in `datasets/mnist_data/data_source_link`

**Penn TreeBank** : dataset can be downloaded from link in `datasets/dl4mt/data_source_link`

`build_dictionary.py` can be used to build a word-level dictionary.

## Running the code

Training (+ validation/testing) can be performed by running `run_rnn.py`.

Check `run_rnn.py --help` for an extensive list of arguments.

This code supports a regular RNN and a factorized RNN with a configurable spectral margin, as well as a wide choice of transition matrix initializations and hidden-to-hidden transition nonlinearities. This code does not include an LSTM setup at this time.

Refactoring code for release may have introduced some bugs - please file an issue if you find any.
