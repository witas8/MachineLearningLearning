import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

DATADIR = "C:/Users/witko/PycharmProjects/LoadOwnData/Pets"
CATEGORIS = ["Dog", "Cat"]

for category in CATEGORIS:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # img_array = pixels
        plt.imshow(img_array, cmap="gray")
        #plt.show()
        break
    break

# normalize images to have the same shape = the same image size
IMG_SIZE = 50
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap="gray")
#plt.show()


# training data
training_data = []


def create_training_data():
    for category in CATEGORIS:
        path = os.path.join(DATADIR, category)
        # map features to the labels as dog = 0, cat =  1 as categories were defined -> CATEGORIS = ["Dog", "Cat"]
        class_num = CATEGORIS.index(category)
        IMG_SIZE = 50
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # img_array = pixels
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                print('Good frame!')
            except cv2.error as e:
                print('Invalid frame!')


create_training_data()
print(len(training_data))

# the same number of dogs and cats images 50/50 to keep balance in model accuracy
random.shuffle(training_data)  # training data is a list (mutable) training_data=(image_array, label)

# features, labels
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
# always we have to do reshape
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # -1 = any number. 1 = gray scale


# save create training data
pickle_out = open("X_pets.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_pets.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# open saved training data - features X and labels Y
X = pickle.load(open("X_pets.pickle", "rb"))
y = pickle.load(open("y_pets.pickle", "rb"))

# convolution neural network
model = Sequential()

# scale data to have values from 0 to 1. For image data we know that max is 255
X = X/255.0

# first input layer
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# third layer
model.add(Flatten())
model.add(Dense(64))

# fourth output layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

# the loss function can be categorical, but the output is 0 or 1 so binary
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# batch_size = how many time we want to pass data
model.fit(X, y, batch_size=32, epochs=5, validation_split=0.1)


