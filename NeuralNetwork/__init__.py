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
    
        hidden = Matrix.matrix_product(self.weights_ih, inputs)
        hidden._element_wise_sum(self.bias_h)
        final_hidden = Matrix.apply_func(hidden, sigmoid)
                
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
        return activated_outputs[-1]