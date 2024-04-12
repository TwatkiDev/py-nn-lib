from NeuralNetwork.NeuralNetwork import NeuralNetwork
from random import shuffle
import sys

def prepare_data(table):
    shuffle(table)
    output = []
    for i in range(len(table)):
        row = table[i].split(',')
        data = {
            "label": int(row[0])
        }

        targets = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        targets[data["label"]] = [1]

        inputs = []
        for c in row[1:]:
            inputs.append([int(c)/255])
        data["targets"] = targets
        data["inputs"] = inputs
        output.append(data)

    return output

if __name__ == '__main__':
    print("start")  

    print("load table")
    train_table = open("mnist_train.csv", "r").readlines()[:6000]
    training_data = prepare_data(train_table)

    test_table = open("mnist_test.csv", "r").readlines()[:2000]
    test_data = prepare_data(test_table)

    print("table loaded")
    nn = NeuralNetwork([784, 10], 0.2)
    
    print("start training")
    for i in range(10):
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