import numpy as np
import mnist  # Get data set from
from keras.models import Sequential  # ANN architecture = Artificial Neural Network
from keras.layers import Dense  # The layers in the ANN
from keras.utils import to_categorical
import matplotlib.pyplot as plt  # Graph

train_images = mnist.train_images()  # training data of images
train_labels = mnist.train_labels()  # training data of the labels
test_images = mnist.test_images()  # testing data images
test_labels = mnist.test_labels()   # testing data labels

# normalize pixels from [0,255] to [-0.5, 0,5] to make easier training
train_images = (train_images/255) - 0.5  # minimum, because max of train_images/255=1
test_images = (test_images/255) - 0.5

# flatten the images from 28x28 dim vector to 784 dimensional vector
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# create the model:
# the goal is to classify the images and learn program digits from images
# model with 3 layers ( 2 with 64 neurons and relu fcn + 1 layer with 10 neuron softmax fcn)
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the model:
# loss fcn = measures how well the model did on training and then tries to improve by using an optimizer
# categorical_crossentropy for classes that are more then 3
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# train the model:
# train labels give us 10 dim vector, so use to_categorical
# epochs = number of iterations over the whole data set to train on
# batch_size = number of samples per gradient update for training
model.fit(train_images, to_categorical(train_labels), epochs=5, batch_size=32)

# evaluate the model:
model.evaluate(test_images, to_categorical(test_labels))

# save the model:
# model.save_weights('model.h5)

# predict on the first 5 test images
n = 10  # how many number to predict from the data set
predictions = model.predict(test_images[:n])
# print(predictions)  # to see probabilities #print(np.argmax(predictions, axis=1))
print(test_labels[:10])  # what are we gonna to predict

#plotting images
for i in range(0, n):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels)  # plt.imshow(pixels, cmap='gray') #to see black and white images
    plt.show()
