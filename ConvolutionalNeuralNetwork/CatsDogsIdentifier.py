import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf  # install tensorflow=1.14
import matplotlib.pyplot as plt


TRAIN_DIR = 'C:/Users/witko/PycharmProjects/ConvolutionalNeuralNetwork/train'
TEST_DIR = 'C:/Users/witko/PycharmProjects/ConvolutionalNeuralNetwork/test'
IMG_SIZE = 50  # to make each picture a perfect square = setting a resolution
LR = 1e-3  # learning rate

# to save model
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


# features convert to grayscale 2D integer array; labels convert to list [1,0] where dog will be 0, cat 1
def label_img(img):
    # dog.XX where XX is a number
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


train_data = create_training_data()  # 2500 labeled images


def process_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])  # img_num is an ID, "labels" of test data are only numbers

    np.save('test_data.npy', testing_data)
    return testing_data


# train_data = np.load('train_data.npy')  # if I gathered data -> train_data = create_training_data()


# create the 2 layered Convolution Neural Network (CNN) with a fully connected layer, and then the output layer
# TIP: 1 layer for linear problem, 6 layer for nonlinear problem (here 5 layers - 2 convolutions and 1 fully connected)
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

# create a model
model = tflearn.DNN(convnet, tensorboard_dir='log')  # DNN = Deep Neural Network

# if os.path.exists('{}.meta'.format(MODEL_NAME)):
#     model.load(MODEL_NAME)
#     print('model loaded!')

# create train and test data
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # -1 = any number. 1 = gray scale
y = [i[1] for i in train]  # label [0,1] = [number, string (label)]

test_X = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

# train a model
model.fit({'input': X}, {'targets': y}, n_epoch=5, validation_set=({'input': test_X}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# # if we are happy with the result the save the model
# model.save(MODEL_NAME)

# visualize
fig = plt.figure()

test_data = process_testing_data()

for num, data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]

    img_data = data[0]
    img_num = data[1]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


# # If you are happy with it, then:
# with open('submission_file.csv', 'w') as f:
#     f.write('id,label\n')
#
# with open('submission_file.csv', 'a') as f:
#     for data in tqdm(test_data):
#         img_num = data[1]
#         img_data = data[0]
#         orig = img_data
#         data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
#         model_out = model.predict([data])[0]
#         f.write('{},{}\n'.format(img_num, model_out[1]))