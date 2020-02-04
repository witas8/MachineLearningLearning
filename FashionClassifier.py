import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt

# load images fashion data set with 6000 images
data = keras.datasets.fashion_mnist

# about 80% of data will be for training and rest for testing
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# labels from 0 - 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# to operate with numbers from 0 - 1, no from 0 to - 255
train_images = train_images/255.0
test_images = test_images/255.0

# to show picture on the screen use: plt.imshow(train_images[7], cmap=plt.cm.binary)  plt.show()
# to show 28x28 pixel representation where 0 - white, 255 - black use: print(train_images[7])


# create the model to classify items (fashion example):
# dense = fully connected layer (number of neurons, activation method)
# softmax = probability of the network that it is a certain value
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

# adam and sparse... = standards, accuracy = what we are looking for and how low we can get our loss to be
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# train the model:
# epochs = how many times our model is gonna see the train data. More epochs can make the model less reliable
model.fit(train_images, train_labels, epochs=8)


# evaluate the model: test_loss, test_acc = model.evaluate(test_images, test_labels)  print("Tested Accuracy:" test_acc)


# making predictions (take the neron that is the largest value and give us the index of that neuron:
# the variable "predict" is a list, so predict[0] is the first item in this list. This is what you want to predict
prediction = model.predict(test_images)
# print(class_names[np.argmax(prediction[0])])

# evaluate as the human if the prediction works well:
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
