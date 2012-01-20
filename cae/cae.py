#!/usr/bin/env python
# encoding: utf-8
"""
cae.py

A pythonic library for Contractive Auto-Encoders. This is
for people who want to give CAEs a quick try and for people
who want to understand how they are implemented. For this
purpose I tried to make the code as simple and clean as possible.
The only dependency is numpy, which is used to perform all
expensive operations. The code is quite fast, however much better
performance can be achieved using the Theano version of this code.

Created by Yann N. Dauphin, Salah Rifai on 2012-01-17.
Copyright (c) 2012 Yann N. Dauphin, Salah Rifai. All rights reserved.
"""

import sys
import os

import numpy


class CAE(object):
    """
    A Contrative Auto-Encoder (CAE) with sigmoid input units and sigmoid
    hidden units.
    """
    def __init__(self, n_hiddens=1024,
                       W=None,
                       c=None,
                       b=None,
                       learning_rate=0.1,
                       batch_size=10,
                       epochs=20):
        """
        Initialize an RBM.
        
        Parameters
        ----------
        n_hiddens : int, optional
            Number of binary hidden units
        W : array-like, shape (n_visibles, n_hiddens), optional
            Weight matrix, where n_visibles in the number of visible
            units and n_hiddens is the number of hidden units.
        c : array-like, shape (n_hiddens,), optional
            Biases of the hidden units
        b : array-like, shape (n_visibles,), optional
            Biases of the visible units
        learning_rate : float, optional
            Learning rate to use during learning
        batch_size : int, optional
            Number of examples to use per gradient update
        epochs : int, optional
            Number of epochs to perform during learning
        """
        self.n_hiddens = n_hiddens
        self.W = W
        self.c = c
        self.b = b
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
    
    def _sigmoid(self, x):
        """
        Implements the logistic function.
        
        Parameters
        ----------
        x: array-like, shape (M, N)

        Returns
        -------
        x_new: array-like, shape (M, N)
        """
        return 1. / (1. + numpy.exp(-x)) 
    
    def encode(self, x):
        """
        Computes the hidden code for the input {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_samples, n_inputs)

        Returns
        -------
        h: array-like, shape (n_samples, n_hiddens)
        """
        return self._sigmoid(numpy.dot(v, self.W) + self.c)
    
    def decode(self, h):
        """
        Compute the reconstruction from the hidden code {\bf h}.
        
        Parameters
        ----------
        h: array-like, shape (n_samples, n_hiddens)
        
        Returns
        -------
        v: array-like, shape (n_samples, n_inputs)
        """
        return self._sigmoid(numpy.dot(h, self.W.T) + self.b)
    
    def reconstruct(self, x):
        """
        Compute the reconstruction of the input {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_samples, n_inputs)
        
        Returns
        -------
        x_new: array-like, shape (n_samples, n_inputs)
        """
        return self.decode(self.encode(x))
    
    def jacobian(self, x):
        """
        Compute jacobian of {\bf h} with respect to {\bf x}.
        
        Parameters
        ----------
        x: array-like, shape (n_samples, n_inputs)
        
        Returns
        -------
        jacobian: array-like, shape (n_samples, n_hiddens, n_inputs)
        """
        h = self.encode(x)
        
        return (h * (1 - h))[:, numpy.newaxis, :] * self.W[numpy.newaxis, : , :]
    
    def loss(self, x):
        """
        Computes the energy of {x} which is the 
        Parameters
        ----------
        v: array-like, shape (n_samples, n_inputs)
        
        Returns
        -------
        free_energy: array-like, shape (n_samples,)
        """
        z = self.reconstruct(x)
        
        j = self.jacobian(x)
        
        return x * numpy.log(z) + (1 - x) * numpy.log(1 - z) + (j**2).mean(1).mean(1)
    
    def _fit(self, v_pos):
        """
        Adjust the parameters to maximize the likelihood of {\bf v}
        using Stochastic Maximum Likelihood (SML).
        
        Parameters
        ----------
        v_pos: array-like, shape (n_samples, n_visibles)
        """
        pass
    
    def fit(self, X, verbose=False):
        """
        Fit the model to the data X.
        
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        """
        if self.W == None:
            self.W = numpy.random.uniform(
                low=-4*numpy.sqrt(6./(X.shape[1]+self.n_hiddens)),
                high=4*numpy.sqrt(6./(X.shape[1]+self.n_hiddens)),
                size=(X.shape[1], self.n_hiddens))
            self.c = numpy.zeros(self.n_hiddens)
            self.b = numpy.zeros(X.shape[1])
            self.h_samples = numpy.zeros((self.n_samples, self.n_hiddens))
        
        inds = range(X.shape[0])
        
        numpy.random.shuffle(inds)
        
        n_batches = len(inds) / self.n_samples
        
        for epoch in range(self.epochs):
            for minibatch in range(n_batches):
                self._fit(X[inds[minibatch::n_batches]])
            
            if verbose:
                loss = self.loss(X).mean()
            
                print "Epoch %d, Loss = %.2f" % (epoch, loss)


def main():
    pass


if __name__ == '__main__':
    main()

