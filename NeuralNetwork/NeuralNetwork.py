from Matrix.NumpyMatrix import NumpyMatrix as Matrix
from NeuralNetwork.Layer import Layer
import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate = 0.1):
        self.layers = []
        self.learning_rate = learning_rate
    
    def __str__(self):
        ret = "[\n"
        for w in self.weights:
            ret += str(w)+"\n"
        ret += "]\n"
        return ret
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_forward(self, inputs_array):
        inputs = Matrix.from_array(inputs_array)
        outputs = self._feed_forward(inputs)
        return Matrix.to_array(outputs)
    
    def _feed_forward(self, inputs):
        outputs = inputs
        for l in self.layers:
            outputs = l.feed_forward(outputs)

        return outputs
    
    def train(self, inputs_array, targets_array):
        inputs = Matrix.from_array(inputs_array)
        targets = Matrix.from_array(targets_array)
        
        outputs = self._feed_forward(inputs)

        err = Matrix.element_wise_sub(targets, outputs)
        
        for l in reversed(self.layers):
            err = l.backward_propagation(err, self.learning_rate)

        return Matrix.to_array(outputs)