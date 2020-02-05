import tensorflow as tf
from tensorflow.python import keras
# download correct version of numpy as install numpy==1.16.1 !!!!!!!!
import numpy as np

# to avoid AVX2 error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# example use saving training models in a checkpoint if the process is big (growth of num_words and embedding words)

# download data as movie reviews from imdb website
data = keras.datasets.imdb

# the variable num_words tells us what type of text are we going to process, because to less words is going to ignore
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=8800)

# display data, where the numbers point a certain word
print(train_data)

# map the words
word_index = data.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}  # k=key=word (3 special characters), v=value=integer
word_index["<PAD>"] = 0  # padding to set movie to the same length and the length will be the longest one to do not cut
word_index["START>"] = 1
word_index["<UNK>"] = 2  # unknown characters
word_index["<UNUSED>"] = 3

# make the values point the strings, so on the screen will be the words. Dict = dictionary (not a list)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# define length to all our data as max number of strings
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# function to decode the training and testing data into readable words
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# print 2 tested data. Choose test data by providing any number to the list [X]  print(decode_review(test_data[0]))
print(len(test_data[0]), len(test_data[1]))


# ############################### commend after saving
# create model to check if the review is good or bad:
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))  # check the similarity between words. 10000 word vectors, 16 dimensions
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))  # squish output neuron should be 0 or 1:

model.summary()


# train the model (loss function calculate a diff between sigma output (from 0 to 1) and binary representation (0 or 1)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# split the training data into 2 sets - validation (check how well our model)
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# batch = how many input data (movies) we are going to load at once, we are not able to feed the model with whole data
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)


# h5 is an extension for a saved model in a tensorflow library
model.save("reviewModel.h5")


# ##################################### use after saving
# after training many models, pick the best one and save it, then load by below line (before commend create model code)
model = keras.models.load_model("reviewModel.h5")


def review_encode(string):
    encoded = [1]  # 1, because <START> is assigned to 1
    # look at the number associated with the words and them to encoded list
    for word in string:
        if word in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)  # 2, because <UNK> unknown is assigned to 2

    return encoded


# use any text file and upload it to your saved model
# download it from -> https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
with open("test_negative_review.txt", encoding="utf-8") as f:
    for line in f.readlines():
        # delete ,.()  because they are symbols, not words
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("\"", "").strip().split(" ")
        # encode and trim data to n words, what was already defined as 250
        encode = review_encode(nline)
        # similar line, but pass the list of encodes to get list of list
        encode = train_data = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"],
                                                                         padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])  # the result is: 1 = positive review, 0 = negative review

