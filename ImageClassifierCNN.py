import tensorflow as tf
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # load data
# x_train.shape = (50000,32,32,3) = (rows, pixel x pixel resolution columns, depth), y_train (50000,1) = (rows, columns)
# to show the first image: x_train[0], 0 = index. result RGB for each pixel ex. 59, 62, 120 -> 1 pixel
# to show picture img: plt.imshow(x_train[0])
# to print label: y_train[0], index in this case 0-9

# One-Hot Endcoding: convert the labels into a set of 10 numbers to input into the neural network
# One-Hot Endcoding = method to determine which neuron has the highest number = chosen label (0-9)
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
# print(y_train_one_hot)  # 10 digit and 1 points the exactly label

# normalize pixels to be values between 0 and 1
x_train = x_train/255
x_test = x_test/255


# CREATE THE CNN MODEL
# create the architecture of the CNN with convolution layer
model = Sequential()

# input layer
# feature maps: get 32 channels to split out, kernel, activation fcn = relu (rectifier (prostownik) linear unit)
# it's a first layer so give an input as 32 by 32 pixels and 3 depth
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))
# create max pooling layer = gets the max element from the convolve (zwijać) features
# pool_size = filter, so from 32 to 32 image the filter will reduce to 16 by 16
model.add(MaxPooling2D(pool_size=(2, 2)))

# second layer
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# flatten makes the image a linear rate = 1D array = 1D vector to feed (connect to) neural network
model.add(Flatten())

# neural network with 1000 neurons
model.add(Dense(1000, activation='relu'))

# output layer
model.add(Dense(10, activation='softmax'))

# compile the model
# categorical_crossentropy for classes grater then 2 (here 10 classes 0-9)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# TRAIN (fit) THE MODEL
# batch_size = total number of training examples
# epochs = number of iterations when the entire data set is passed forward and backward through the neural network
# validation_split 30%, so 70% for training
myModel = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.3)

# get the model accuracy
model.evaluate(x_test, y_test_one_hot)[1]

# visualize the model's accuracy
plt.plot(myModel.history['accuracy'])
plt.plot(myModel.history['val_accuracy'])  # validation accuracy
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')  # loc = screen localization
plt.show()

# visualize the model's loss
plt.plot(myModel.history['loss'])
plt.plot(myModel.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper right')  # loc = screen localization
plt.show()