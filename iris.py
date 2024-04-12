from NeuralNetwork.NeuralNetwork import NeuralNetwork
from random import shuffle
import sys

if __name__ == '__main__':
    print("start")  

    print("load table")
    table = open("iris.csv", "r").readlines()[1:]
    labels = {
        "virginica": 0, 
        "versicolor": 1, 
        "setosa": 2
    }
    shuffle(table)
    
    iris_data = []
    for i in range(len(table)):
        row = table[i].split(',')
        data = {
            "label": labels[row[4][:-1]]
        }

        targets = [[0], [0], [0]]
        targets[data["label"]] = [1]

        inputs = []
        for c in row[:4]:
            inputs.append([float(c)])
        data["targets"] = targets
        data["inputs"] = inputs
        iris_data.append(data)

    training_data = iris_data[:int(len(iris_data)*2/3)]
    test_data = iris_data[int(len(iris_data)*2/3):]

    print("table loaded")
    nn = NeuralNetwork([4, 5, 3], 0.1)
    
    print("start training")
    for i in range(100):
        _training_data = training_data[:]
        shuffle(_training_data)
        total = 0
        for j in range(len(training_data)):
            data = _training_data[j]
            outputs = nn.train(data["inputs"], data["targets"])
            max_idx = outputs.argmax()
            if max_idx == data["label"]:
                total += 1
        print("epoch n{} average = {}".format(i, total/len(training_data)))
        
    print("end training")
    
    print("start testing")
    total = 0
    for data in test_data:
        outputs = nn.feed_forward(data["inputs"])
        max_idx = outputs.argmax()
        if max_idx == data["label"]:
            total += 1
    
    print("average = {}".format(total/len(test_data)))

    print("end testing")
    
    print("end")