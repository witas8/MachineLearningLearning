# add " buying,maint,door,persons,lug_boot,safety, class " at the top of car.data to use pandas
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# to operate with data let's convert it to the numerical one
labels = preprocessing.LabelEncoder()
buying = labels.fit_transform(list(data["buying"]))  # convert buying column to the integer list
maint = labels.fit_transform(list(data["maint"]))
door = labels.fit_transform(list(data["door"]))
persons = labels.fit_transform(list(data["persons"]))
lug_boot = labels.fit_transform(list(data["lug_boot"]))
safety = labels.fit_transform(list(data["safety"]))
classes = labels.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(classes)  # labels

# do not use to much test data, because for training will not be enough
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# knn = classification algorithm where k = amount of neighbours to group for (ex. k=3)
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

# predict data
predicted = model.predict(x_test)

# what are the data points, prediction and the actual values
names = ["unacc", "acc", "good", "vgood"]  # 0-3 where unacc = 0, vgood = 3
for i in range(len(x_test)):
    print("Predicted data: ", names[predicted[i]], "Data: ", x_test[i], "Actual: ", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 9, True)  # 9 = number of neighbors
    print("N: ", n)  # print neighbors with array [distance between all neighbors, index of neighbors]
