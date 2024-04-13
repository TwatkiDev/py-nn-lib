from NeuralNetwork.NeuralNetwork import NeuralNetwork
from random import shuffle
from NeuralNetwork.Layer import Layer, SIGMOID, SOFTMAX

if __name__ == '__main__':
    print("start")  
    training_data = [
        {
            "inputs": [0, 0], 
            "targets": [0]
        }, 
        {
            "inputs": [1, 0], 
            "targets": [1]
        }, 
        {
            "inputs": [0, 1], 
            "targets": [1]
        }, 
        {
            "inputs": [1, 1], 
            "targets": [0]
        }
    ]
    nn = NeuralNetwork(0.1)
    nn.add_layer(Layer(2, 10, SIGMOID))
    nn.add_layer(Layer(10, 1, SIGMOID))
    
    print("start training")
    for i in range(2000):
        _training_data = training_data[:]
        shuffle(_training_data)
        total = 0
        for j in range(0, len(training_data), 4):
            inputs_batch = [training_data[0]["inputs"], training_data[1]["inputs"], training_data[2]["inputs"], training_data[3]["inputs"]]
            targets_batch = [training_data[0]["targets"], training_data[1]["targets"], training_data[2]["targets"], training_data[3]["targets"]]
            outputs = nn.train(inputs_batch, targets_batch)
            for h in range(len(outputs)):
                o = outputs[h]
                if round(o[0]) == targets_batch[h][0]:
                    total += 1
        print("epoch n{} average = {}".format(i, total/len(training_data)))
    
    print("start testing")
    print(nn.feed_forward([[0, 0]]))
    print(nn.feed_forward([[0, 1]]))
    print(nn.feed_forward([[1, 0]]))
    print(nn.feed_forward([[1, 1]]))