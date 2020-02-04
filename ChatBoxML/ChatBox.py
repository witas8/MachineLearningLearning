import nltk
# nltk.download()
from nltk.stem.lancaster import LancasterStemmer
import numpy
import json
import tflearn
import random
import tensorflow  # pip install tensorflow==1.13.2
import pickle
stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)
# print(data["intents"])

try:
    with open("data.pickle", "rb") as f:  # commend to train the model again
        words, labels, training, output = pickle.load(f)  # save to pickle file  # commend to train the model again
except:  # commend to train the model again
    words = []  # blank list
    labels = []
    docs_x = []
    docs_y = []

    # stemming = take each word tak is in a pattern and bring it to the root word (whats->what, there?->there, hi!->hi)
    # eliminate extra characters to train data -> model more accurate
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)  # add tokens to words list
            docs_x.append(tokens)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]  # words list corresponds with nltk library
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # create training and testing output
    # now we have strings, but networks understand only numbers
    # input = list of existing words 1 and not existing 0 for tags
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        tokens = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in tokens:
                bag.append(1)  # yes this word exist
            else:
                bag.append(0)  # no this word does not exist

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1  # check in labels list where the tag is, then set this value to 1 in output

        training.append(bag)  # input
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:  # commend to train the model again
        pickle.dump((words, labels, training, output), f)  # commend to train the model again

########################################## except tab till this position

# create neural network model called net (fully connected = all neurons are connected with each other in every layer)
tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])  # define expecting input length
net = tflearn.fully_connected(net, 8)  # 8 neurons for hidden layer
net = tflearn.fully_connected(net, 8)  # 8 neurons for hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # 6 tags = 6 neurons for OUTPUT layer
net = tflearn.regression(net)


# softmax gives a probability to point which output tag can be concerned
# train the model that predicts what tag give to the user
model = tflearn.DNN(net)

try:  # commend to train the model again
    model.load("model.tflearn")  # commend to train the model again
except:  # commend to train the model again
    # epochs = how many times program is gonna to see the same data (play with)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


# make predictions:
def bag_of_words(string, words):
    bag = [0 for _ in range(len(words))]  # bland bag of words list
    s_words = nltk.word_tokenize(string)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s in s_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  # else will be zeros

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quite to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        # print(results) bunch of probabilities to show how the program thinks = neuron consider words with probability
        results_index = numpy.argmax(results)
        tag = labels[results_index]  # print(tag)

        if results[results_index] > 0.7:
            for tagger in data["intents"]:
                if tagger['tag'] == tag:
                    responses = tagger['responses']

            print(random.choice(responses))
        else:
            print("Sorry, I do not understand. Can you ask me again?")


chat()
