"""
network2.py
-----------

An improved version of network.py
New implementations:
    - stochastic gradient descent learning algorithm
      for a feedforward neural network
    - addition of the cross-entropy cost function
    - regularization
    - better initialization of network weights
"""

import json
import random
import sys

import numpy as np

class QuadratiCost(object):
    
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output 'a' and desired output 'y'."""
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost accosicated with an output 'a' and desired output 'y'.
        
        Note:   np.nan_to_num is used to ensure numerical stability.  
                In particular, if both 'a' and 'y' have a 1.0 in the same slot, then
                the expression (1-y)*np.log(1-a) returns nan. 
                The np.nan_to_num ensures that it is converted to correct value 0.0

        """
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.
        
        Note:   the parameter 'z' is not used by the method.
                It is included in order to make the interface consistent
                w/ the delat method for other cost classes.
        """

        return (a-y)