from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.Layer import Layer, SIGMOID, SOFTMAX
from random import shuffle
import sys

labels = {
    "virginica": 0, 
    "versicolor": 1, 
    "setosa": 2
}

def get_batch(dataset, from_idx, batch_size):
    out = {
        "labels": [],
        "inputs": [],
        "targets": []
    }
    for i in range(from_idx, min(len(dataset), from_idx+batch_size)):
        data = dataset[i]
        out["labels"].append(data["label"])
        out["inputs"].append(data["inputs"][0])
        out["targets"].append(data["targets"][0])

    return out

def prepare_data(table):
    shuffle(table)
    output = []
    for i in range(len(table)):
        row = table[i].split(',')
        data = {
            "label": labels[row[-1][:-1]]
        }

        targets = [0, 0, 0]
        targets[data["label"]] = 1

        inputs = []
        for c in row[:-1]:
            inputs.append(float(c))
        data["targets"] = [targets]
        data["inputs"] = [inputs]
        output.append(data)

    return output

if __name__ == '__main__':
    print("start")  

    print("load table")
    table = open("iris.csv", "r").readlines()[1:]
    shuffle(table)

    iris_data = prepare_data(table)

    training_data = iris_data[:int(len(iris_data)*2/3)]
    test_data = iris_data[int(len(iris_data)*2/3):]

    print("table loaded")
    nn = NeuralNetwork(0.05)
    nn.add_layer(Layer(4, 5, SIGMOID))
    nn.add_layer(Layer(5, 3, SIGMOID))
    
    print("start training")
    for i in range(1000):
        _training_data = training_data[:]
        shuffle(_training_data)
        total = 0
        batch_size = 32
        for j in range(0, len(training_data), batch_size):
            batch = get_batch(training_data, j, batch_size)
            outputs = nn.train(batch["inputs"], batch["targets"])
            for h in range(len(outputs)):
                o = outputs[h]
                max_idx = o.argmax()
                if max_idx == batch["labels"][h]:
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

    