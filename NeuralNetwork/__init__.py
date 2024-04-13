from Matrix import Matrix
from math import e

def sigmoid(x):
    return 1 / (1 + e**-x)

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, layers, learning_rate = 0.1):
        self.layers = layers
        
        self.weights = []
        self.biases = []
        for i in range(1, len(layers)):
            wi = Matrix(layers[i], layers[i-1])
            wi.randomize()
            self.weights.append(wi)
            
            bi = Matrix(layers[i], 1)
            bi.randomize()
            self.biases.append(bi)
        
        self.learning_rate = learning_rate
    
    def feed_forward(self, inputs_array):
        inputs = Matrix.from_array(inputs_array)
        _, activated_outputs = self._feed_forward(inputs)
        return activated_outputs[-1]

    def _feed_forward(self, inputs):   
        layers_outputs = [inputs]
        activated_layers_outputs = [Matrix.apply_func(inputs, sigmoid)]
        
        for i in range(0, len(self.weights)):
            #Hi = sigmoid(Wi*H(i-1)+Bi) 
            outputs = Matrix.matrix_product(self.weights[i], activated_layers_outputs[-1])
            outputs._element_wise_sum(self.biases[i])
            layers_outputs.append(Matrix.copy(outputs))
            outputs._apply_func(sigmoid) 
            activated_layers_outputs.append(outputs)
            
        return layers_outputs, activated_layers_outputs
    
    def train(self, inputs_array, targets_array):
        inputs = Matrix.from_array(inputs_array)
        targets = Matrix.from_array(targets_array)
        
        errors = []
        
        outputs, activated_outputs = self._feed_forward(inputs)
        
        err = Matrix.element_wise_sub(targets, activated_outputs[-1])
        errors.append(err)
        
        for i in reversed(range(1, len(self.weights))):
            Wi_T = Matrix.transpose(self.weights[i])
            err = Matrix.matrix_product(Wi_T, errors[len(errors)-1])
            errors.append(err)
            
        errors.reverse()
        
        for i in reversed(range(0, len(errors))):
            gradients = Matrix.apply_func(outputs[i+1], dsigmoid)
            gradients._element_wise_mul(errors[i])
            gradients._scalar_product(self.learning_rate)
                        
            activated_output_i_T = Matrix.transpose(activated_outputs[i])
            weights_deltas = Matrix.matrix_product(gradients, activated_output_i_T)
            self.weights[i]._element_wise_sum(weights_deltas)
            self.biases[i]._element_wise_sum(gradients)