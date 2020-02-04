# get big data set and minimize the data to the valuable information

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

# I do not want to use 32 attributes, so I have chosen what I want.  print(data.head()) - to check attributes and values
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# choose what you want to predict - I have chosen G3. label = what you want to get, so G3 = label
predict = "G3"
x = np.array(data.drop([predict], 1))  # new data frame = training data
y = np.array(data[predict])

# split to 4 different arrays: x test, y test, x train, y train. Test for test accuracy
# test_size = split 10% data into test samples and every time different samples are chosen to train
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# ################## commend after loading pickle (start)
best = 0
times = 30
for _ in range(times):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)  # predict what a student grade is going to be at the end of the year

    if accuracy > best:
        best = accuracy  # save a new model if current score is better (best) then previous (accuracy)
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)  # pickle saves model
# ############# commend after loading pickle (end)


# load the saved pickle
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# ax+b
print("Coefficient: ", linear.coef_)  # a -> a0*x+a1*y+a2*z+a3*p+a4*u
print("Intercept: ", linear.intercept_)  # b

predictions = linear.predict(x_test)  # prediction = actual grade based on values of attributes, x_test = array in array
# print the final grade (G3) prediction based on the rest already defined attributes (printed values as [1, 2, 3, 4, 5])
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])  # first sem grade, second sem grade, hours of study, failures, absences

p = "G1"  # attribute, G3 - label
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()  # for 600 students