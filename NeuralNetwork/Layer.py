from Matrix.NumpyMatrix import NumpyMatrix as Matrix
from Matrix.SimpleMatrix import SimpleMatrix # as Matrix
import numpy as np

def sigmoid(values):
    return 1 / (1 + np.exp(-1*values))

def dsigmoid(values):
    sig_values = sigmoid(values)
    return sig_values * (1 - sig_values)

def softmax(values):
    return np.exp(values) / np.sum(np.exp(values))

def dsoftmax(values):
    e_x = np.exp(values)
    y = np.sum(np.exp(values))-np.exp(values)
    return y*e_x/((e_x+y)**2)

SIGMOID = "sigmoid"
SOFTMAX = "softmax"

activation_functions = {
    "sigmoid": (sigmoid, dsigmoid),
    "softmax": (softmax, dsoftmax)
}

class Layer:
    def __init__(self, inputs, outputs, activation_function="sigmoid"):
        self.weights = Matrix(inputs, outputs)
        self.bias = Matrix(1, outputs)

        self.weights.randomize()
        self.bias.randomize()

        self.act_func, self.d_act_func = activation_functions[activation_function]

    def feed_forward(self, inputs):
        self.inputs = inputs

        #Hi = act_func(Wi*H(i-1)+Bi) 
        outputs = Matrix.matrix_product(inputs, self.weights)
        for i in range(len(outputs.matrix)):
            tmp = Matrix.from_array([outputs.matrix[i]])
            tmp = Matrix.element_wise_sum(tmp, self.bias)
            outputs.matrix[i] = Matrix.to_array(tmp)
        self.outputs = Matrix.copy(outputs)
        outputs = Matrix.from_array(self.act_func(Matrix.to_array(outputs)))
        self.activated_outputs = Matrix.copy(outputs)

        return self.activated_outputs

    def backward_propagation(self, output_error, learning_rate):
        Wi_T = Matrix.transpose(self.weights)
        err = Matrix.matrix_product(output_error, Wi_T)
        
        gradients = Matrix.apply_func(self.outputs, self.d_act_func)
        gradients = Matrix.element_wise_mul(gradients, output_error)
        gradients = Matrix.scalar_product(gradients, learning_rate)
        
        activated_inputs_T = Matrix.transpose(self.inputs)
        weights_deltas = Matrix.matrix_product(activated_inputs_T, gradients)
        self.weights = Matrix.element_wise_sum(weights_deltas, self.weights)

        gradients = Matrix.from_array([gradients.matrix.sum(axis=0)])
        self.bias = Matrix.element_wise_sum(self.bias, gradients)

        return err