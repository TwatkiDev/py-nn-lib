from NeuralNetwork import NeuralNetwork
from random import randint, shuffle

if __name__ == '__main__':
    training_data = [
        {
            "inputs": [[0], [0]], 
            "targets": [[0]]
        }, 
        {
            "inputs": [[1], [0]], 
            "targets": [[1]]
        }, 
        {
            "inputs": [[0], [1]], 
            "targets": [[1]]
        }, 
        {
            "inputs": [[1], [1]], 
            "targets": [[0]]
        }
    ]
    nn = NeuralNetwork(2, 2, 1)

    for _ in range(50000):
        _training_data = training_data[:]
        shuffle(_training_data)
        for data in _training_data:
            nn.train(data["inputs"], data["targets"])

    print(nn.feed_forward([[0], [0]]))
    print(nn.feed_forward([[0], [1]]))
    print(nn.feed_forward([[1], [0]]))
    print(nn.feed_forward([[1], [1]]))