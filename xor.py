from NeuralNetwork.NeuralNetwork import NeuralNetwork
from random import shuffle

if __name__ == '__main__':
    print("start")  
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
    nn = NeuralNetwork([2, 5, 1], 0.3)
    
    print("start training")
    for i in range(2000):
        _training_data = training_data[:]
        shuffle(_training_data)
        total = 0
        for j in range(len(training_data)):
            data = training_data[j]
            outputs = nn.train(data["inputs"], data["targets"])
            if round(list(outputs[0])[0]) == data["targets"][0][0]:
                total += 1
        print("epoch n{} average = {}".format(i, total/len(training_data)))
    
    print("start testing")
    print(nn.feed_forward([[0], [0]]))
    print(nn.feed_forward([[0], [1]]))
    print(nn.feed_forward([[1], [0]]))
    print(nn.feed_forward([[1], [1]]))