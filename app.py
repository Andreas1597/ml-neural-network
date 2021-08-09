"""
app.py
----------

Main module where the neural network is initialzed, trained and tested.
"""

import src.mnist_loader
import src.network

training_data, validation_data, test_data = src.mnist_loader.load_data_wrapper()

net = src.network.Network([784, 30, 10])

net.SGD(training_data=training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
