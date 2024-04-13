from Matrix import Matrix
from math import e

def sigmoid(x):
    return 1 / (1 + e**-x)

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.weights_ih = Matrix(hidden_nodes, input_nodes)
        self.weights_ho = Matrix(output_nodes, hidden_nodes)
        
        self.weights_ih.randomize()
        self.weights_ho.randomize()
        
        self.bias_h = Matrix(hidden_nodes, 1)
        self.bias_o = Matrix(output_nodes, 1)
        
        self.bias_h.randomize()
        self.bias_o.randomize()
        
        self.learning_rate = 0.1
        
    def _feed_forward(self, inputs):   
        layers_outputs = []
        activated_layers_outputs = []
    
        # H = sigmoid(Wih*I+Bh)
        hidden = Matrix.matrix_product(self.weights_ih, inputs)
        hidden._element_wise_sum(self.bias_h)
        final_hidden = Matrix.apply_func(hidden, sigmoid)
                
        # O = sigmoid(Who*H+Bo)
        outputs = Matrix.matrix_product(self.weights_ho, final_hidden)
        outputs._element_wise_sum(self.bias_o)
        final_outputs = Matrix.apply_func(outputs, sigmoid) 

        layers_outputs.append(hidden)
        layers_outputs.append(outputs)

        activated_layers_outputs.append(final_hidden)
        activated_layers_outputs.append(final_outputs)
                                
        return layers_outputs, activated_layers_outputs
    
    def feed_forward(self, inputs_array):
        inputs = Matrix.from_array(inputs_array)
        _, activated_outputs = self._feed_forward(inputs)
        return activated_outputs[1]
    
    def train(self, inputs, targets): 
        inputs = Matrix().from_array(inputs)
        targets = Matrix().from_array(targets)
        
        outputs, activated_outputs = self._feed_forward(inputs)
        
        # errors calculation 
        output_errors = Matrix.element_wise_sub(targets, activated_outputs[1])
        weights_ho_T = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.matrix_product(weights_ho_T, output_errors)

        # gradients calculation for the output layer
        gradients = Matrix.apply_func(outputs[1], dsigmoid)
        gradients._element_wise_mul(output_errors)
        gradients._scalar_product(self.learning_rate)
        
        # update weights between the hidden layer and the output layer and biases of the output layer
        hidden_T = Matrix.transpose(activated_outputs[0])
        weights_ho_deltas = Matrix.matrix_product(gradients, hidden_T)
        self.weights_ho._element_wise_sum(weights_ho_deltas)
        self.bias_o._element_wise_sum(gradients)
    
        # gradients calculation for the hidden layer
        hidden_gradients = Matrix.apply_func(outputs[0], dsigmoid)
        hidden_gradients._element_wise_mul(hidden_errors)
        hidden_gradients._scalar_product(self.learning_rate)
        
        # update weights between the input layer and the hidden layer and biases of the hidden layer
        inputs_T = Matrix.transpose(inputs)
        weights_ih_deltas = Matrix.matrix_product(hidden_gradients, inputs_T)
        self.weights_ih._element_wise_sum(weights_ih_deltas)
        self.bias_h._element_wise_sum(hidden_gradients)