from Matrix.NumpyMatrix import NumpyMatrix as Matrix
from NeuralNetwork.Layer import Layer
import jsonpickle

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
    
    @staticmethod
    def encode(nn):
        return jsonpickle.encode(nn)
    
    @staticmethod
    def decode(pickle):
        return jsonpickle.decode(pickle)
    
    @staticmethod
    def save_to_file(nn, file_name):
        pickle = NeuralNetwork.encode(nn)
        fd = open(file_name, "w")
        fd.write(pickle)
        fd.close()

    @staticmethod
    def load_from_file(file_name):
        fd = open(file_name, "r")
        pickle = fd.read()
        fd.close()
        return NeuralNetwork.decode(pickle)

