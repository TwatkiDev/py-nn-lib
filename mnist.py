from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.Layer import Layer, SIGMOID, SOFTMAX
from random import shuffle
import sys

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
            "label": int(row[0])
        }

        targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        targets[data["label"]] = 1

        inputs = []
        for c in row[1:]:
            inputs.append(int(c)/255)
        data["targets"] = [targets]
        data["inputs"] = [inputs]
        output.append(data)

    return output

if __name__ == '__main__':
    print("start")  

    print("load table")
    train_table = open("mnist_train.csv", "r").readlines()
    training_data = prepare_data(train_table)

    test_table = open("mnist_test.csv", "r").readlines()
    test_data = prepare_data(test_table)

    print("table loaded")
    nn = NeuralNetwork(0.2)
    nn.add_layer(Layer(784, 50, SIGMOID))
    nn.add_layer(Layer(50, 10, SOFTMAX))

    print("start training")
    for i in range(10):
        _training_data = training_data[:]
        shuffle(_training_data)
        total = 0
        batch_size = 128
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