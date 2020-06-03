from nn import *

input_node = 784
hidden_node = 64
output_node = 3

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
testing = []


def prepareData(new_data, data, label):
    # for i in range(len(data)):
    #     data[i,]
    # np.random.shuffle(data)

    new_data["train_data"] = data[0:800, :]
    new_data["train_data"] = new_data["train_data"] / 255
    new_data["test_data"] = data[800:1000, :]
    new_data["test_data"] = new_data["test_data"] / 255

    for k in new_data["train_data"]:
        training.append(np.append(k, label))

    for k in new_data["test_data"]:
        testing.append(np.append(k, label))

    # new_data["label"] = label


nn = NeuralNetwork(input_node, hidden_node, output_node)

prepareData(cats, cat_data, 0)
prepareData(rainbows, rainbow_data, 1)
prepareData(trains, train_data, 2)

training = np.array(training)

testing = np.array(testing)


def train(n):
    np.random.shuffle(training)

    for i in range(len(training)):
        inputs = training[i, :784]
        label = int(training[i, 784])
        # print(inputs, len(inputs))
        # print(label)
        target = np.array([0, 0, 0])
        target[label] = 1

        nn.train(inputs, target)

    print("Trained for", n + 1, "epochs.")


for i in range(10):
    train(i)

# ran = random.randint(0,599)
#
# output = testing[ran, 784]
# prediction = nn.predict(testing[ran, :784])
# max_val = np.argmax(prediction)
# print("prediction: ", prediction)
# print("output: ", output)
# print("prediction index: ", max_val)
print("Traing finish, Testing starts.....")

count = 0
total = len(testing)

for i in range(len(testing)):

    output = testing[i, 784]
    prediction = nn.predict(testing[i, :784])
    max_val = np.argmax(prediction)

    if output == max_val:
        count = count + 1

accuracy = (count/total) * 100
print("Accuracy :", accuracy)

