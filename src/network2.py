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

class Network(object):


    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list of 'sizes' contains the number of neurons in the respective
        layers of the network. 

        Example:
            net = Network([2, 3, 1])

            A list of [2, 3, 1] means the network consists of three layers
            with the first layer (input layer) containing 2 neurons, the 
            second layer 3 neurons, and the thirt layer (output layer) 1 neuron.

        The biases and weights for the network are initialized randomly using
            -> self.default_weight_initializer (see docsting for that method)
        
        """        
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.default_weights_initalizeer()
        self.cost = cost

    def default_weights_initalizeer(self):
        """Initialize each weight using a Gaussian distribution
        with mean zero and standard deviation 1 over the square root 
        of the number of weights connecting to the same neuron.

        Initialize the biases using a Gaussian distribution w/
        mean zero and std deviation 1.

        Note:   the first layer is assumed to be an input layer, 
                and by convention we will not set any bias for those
                neurons, since biases are only ever used in computing
                the outputs of later neurons.
        
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weights_initalizeer(self):
        """Initialize each weight using a Gaussian distribution
        with mean zero and standard deviation 1

        Initialize the biases using a Gaussian distribution w/
        mean zero and std deviation 1.

        Note:   the first layer is assumed to be an input layer, 
                and by convention we will not set any bias for those
                neurons, since biases are only ever used in computing
                the outputs of later neurons.
        
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def feedforward(self, a):
        """Return output of the network with input ``a``."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGM(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data = None,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False,
            early_stopping_n = 0):
        """Train the neural network using mini-batch stochastic gradient descent.def
        The ``training_data`` is a list of tuples ``(x, y)`` representing the training inputs
        and the desired outputs. The other non-optional params are self-explanatory.
        
        """
        best_accuracy = 1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality
        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuray = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data)
                )

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuray.append(accuracy)
                print("Accuracy on training data {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data {} / {}".format(accuracy, n_data))

            # early stopping
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    print("Early stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change +=1

                if (no_accuracy_change == early_stopping_n):
                    print("Early stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
            
            return evaluation_cost, evaluation_accuracy, training_cost, training_accuray
            
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network`s weights and biases by applying grad
        descent using backpropagation to a single mini batch. The
        ``mini_batch``is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda``is the regularization param, and
        ``n``is the total size of the training data.
        
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,
                       nb in zip(self.biases, nabla_b)]   

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x. ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations
        zs = []  # list to store all the z vectors per layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
