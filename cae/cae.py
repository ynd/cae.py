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
import pdb
import numpy


class CAE(object):
    """
    A Contrative Auto-Encoder (CAE) with sigmoid input units and sigmoid
    hidden units.
    """
    def __init__(self, 
                 n_hiddens=1024,
                 W=None,
                 c=None,
                 b=None,
                 learning_rate=0.001,
                 jacobi_penalty=0.1,
                 batch_size=10,
                 epochs=200):
        """
        Initialize a CAE.
        
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
        self.jacobi_penalty = jacobi_penalty
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
        return self._sigmoid(numpy.dot(x, self.W) + self.c)
    
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
        Computes the error of the model with respect
        
        to the total cost.
        
        -------
        v: array-like, shape (n_samples, n_inputs)
        
        Returns
        -------
        free_energy: array-like, shape (n_samples,)
        """
        def _reconstruction_loss():
        """
        Computes the error of the model with respect
        
        to the reconstruction (cross-entropy) cost.
        
        """
            z = self.reconstruct(x)
            return (- (x * numpy.log(z) + (1 - x) * numpy.log(1 - z)).sum(1)).mean()

        def _jacobi_loss():
        """
        Computes the error of the model with respect
        
        the Frobenius norm of the jacobian.
        
        """
            j = self.jacobian(x)
            return (j**2).sum(2).sum(1).mean()

        return _reconstruction_loss() + self.jacobi_penalty * _jacobi_loss()
    
    def _fit(self, x):
        """
        TODO
        
        Parameters
        ----------
        x: array-like, shape (n_samples, n_visibles)
        """
        def _fit_contraction():
            """
            Compute the gradient of the contraction cost w.r.t parameters.
            """
            h = self.encode(x)

            a = (h * (1 - h))**2 

            b = x[:,:,numpy.newaxis] * ((1 - 2 * h) * a * (self.W**2).sum(0)[numpy.newaxis,:])[:,numpy.newaxis,:]

            c = a[:,numpy.newaxis,:] * self.W

            return (b+c).mean(0)
            
        def _fit_reconstruction():
            """                                                                 
            Compute the gradient of the reconstruction cost w.r.t parameters.      
            """

            h = self.encode(x)
            r = self.decode(h)

            dedr = -( x/r - (1 - x)/(1 - r) ) 

            a = r*(1-r)
            b = h*(1-h)
            
            od = a * dedr
            oe = b * numpy.dot(od, self.W)

            gW = b * oe

            return gW.mean(0),od.mean(0),oe.mean(0)

        W_rec,b_rec,c_rec = _fit_reconstruction()
        self.W -= self.learning_rate * ((W_rec) + self.jacobi_penalty * _fit_contraction())
        self.b -= self.learning_rate * b_rec 
        self.c -= self.learning_rate * c_rec


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
        
        inds = range(X.shape[0])
        
        numpy.random.shuffle(inds)
        
        n_batches = len(inds) / self.batch_size
        
        for epoch in range(self.epochs):
            for minibatch in range(n_batches):
                self._fit(X[inds[minibatch::n_batches]])
            
            if verbose:
                loss = self.loss(X).mean()
                sys.stdout.flush()
                print "Epoch %d, Loss = %.2f" % (epoch, loss)


def main():
    pass


if __name__ == '__main__':
    main()

