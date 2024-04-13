from Matrix.NumpyMatrix import NumpyMatrix as Matrix
from Matrix.SimpleMatrix import SimpleMatrix # as Matrix
import numpy as np

def sigmoid(values):
    return 1 / (1 + np.exp(-1*values))

def dsigmoid(values):
    sig_values = sigmoid(values)
    return sig_values * (1 - sig_values)

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = Matrix(output_size, input_size)
        self.weights.randomize()

        self.bias = Matrix(output_size, 1)
        self.bias.randomize()

        self.act_func = sigmoid
        self.d_act_func = dsigmoid

    def feed_forward(self, inputs):
        self.inputs = inputs

        #Hi = act_func(Wi*H(i-1)+Bi) 
        outputs = Matrix.matrix_product(self.weights, inputs)
        outputs = Matrix.element_wise_sum(outputs, self.bias)
        self.outputs = Matrix.copy(outputs)
        outputs = Matrix.from_array(self.act_func(Matrix.to_array(outputs)))
        self.activated_outputs = Matrix.copy(outputs)

        return self.activated_outputs

    def backward_propagation(self, output_error, learning_rate):
        Wi_T = Matrix.transpose(self.weights)
        err = Matrix.matrix_product(Wi_T, output_error)

        gradients = Matrix.apply_func(self.outputs, self.d_act_func)
        gradients = Matrix.element_wise_mul(gradients, output_error)
        gradients = Matrix.scalar_product(gradients, learning_rate)
        
        activated_inputs_T = Matrix.transpose(self.inputs)
        weights_deltas = Matrix.matrix_product(gradients, activated_inputs_T)
        self.weights = Matrix.element_wise_sum(self.weights, weights_deltas)
        self.bias = Matrix.element_wise_sum(self.bias, gradients)

        return err