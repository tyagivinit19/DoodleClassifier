from nn import *

CAT = 0
RAINBOW = 1
TRAIN = 2

cat_data = np.load("data/cat1000.npy")
rainbow_data = np.load("data/rainbow1000.npy")
train_data = np.load("data/train1000.npy")

cats = {}
rainbows = {}
trains = {}

training = []


def prepareData(new_data, data, label):
    # for i in range(len(data)):
    #     data[i,]

    new_data["train_data"] = data[0:800, :]
    new_data["test_data"] = data[800:1000, :]

    for i in new_data["train_data"]:
        training.append(np.append(i, label))

    # new_data["label"] = label



prepareData(cats, cat_data, 0)
prepareData(rainbows, rainbow_data, 1)
prepareData(trains, train_data, 2)

training = np.array(training)
np.random.shuffle(training)

# nn = NeuralNetwork(784, 64, 3)
