from Matrix.NumpyMatrix import NumpyMatrix as Matrix
from Matrix.SimpleMatrix import SimpleMatrix # as Matrix
from math import e
import numpy as np

def sigmoid(x):
    return 1 / (1 + e**-x)

def dsigmoid(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)

def softmax_(values):
    exp_values = np.exp(values)
    exp_values_sum = np.sum(exp_values)
    return exp_values/exp_values_sum

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

        self.act_func = sigmoid
        self.d_act_func = dsigmoid
    
    def __str__(self):
        ret = "[\n"
        for w in self.weights:
            ret += str(w)+"\n"
        ret += "]\n"
        return ret
    
    def feed_forward(self, inputs_array):
        inputs = Matrix.from_array(inputs_array)
        outputs, activated_outputs = self._feed_forward(inputs)
        return Matrix.to_array(activated_outputs[-1])
    
    def _feed_forward(self, inputs):   
        layers_outputs = [inputs]
        activated_layers_outputs = [Matrix.apply_func(inputs, self.act_func)]
        
        for i in range(0, len(self.weights)):
            #Hi = act_func(Wi*H(i-1)+Bi) 
            outputs = Matrix.matrix_product(self.weights[i], activated_layers_outputs[-1])
            outputs = Matrix.element_wise_sum(outputs, self.biases[i])
            layers_outputs.append(Matrix.copy(outputs))
            outputs = Matrix.apply_func(outputs, self.act_func) 
            activated_layers_outputs.append(Matrix.copy(outputs))     
        return layers_outputs, activated_layers_outputs
    
    def train(self, inputs_array, targets_array):
        inputs = Matrix.from_array(inputs_array)
        targets = Matrix.from_array(targets_array)
        
        errors = []
        
        outputs, activated_outputs = self._feed_forward(inputs)

        err = Matrix.element_wise_sub(activated_outputs[-1], targets) 
        errors.append(err)
        
        for i in reversed(range(1, len(self.weights))):
            Wi_T = Matrix.transpose(self.weights[i])
            err = Matrix.matrix_product(Wi_T, errors[-1])
            errors.append(err)
            
        errors.reverse()
        
        for i in reversed(range(0, len(errors))):
            gradients = Matrix.apply_func(outputs[i+1], self.d_act_func)
            gradients = Matrix.element_wise_mul(gradients, errors[i])
            gradients = Matrix.scalar_product(gradients, self.learning_rate)
                        
            activated_output_i_T = Matrix.transpose(activated_outputs[i])
            weights_deltas = Matrix.matrix_product(gradients, activated_output_i_T)
            self.weights[i] = Matrix.element_wise_sub(self.weights[i], weights_deltas)
            self.biases[i] = Matrix.element_wise_sub(self.biases[i], gradients)

        return Matrix.to_array(activated_outputs[-1])