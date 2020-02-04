# svm = classifier
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()  # print(cancer.feature_names) print(cancer.target_names)

x = cancer.data
y = cancer.target

# it is not recommended to go higher them 30% of test_size
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train, y_train)  # 0 represents malignant (złośliwy, szkodliwy), 1 represents benign (łagodny)

classes = ['malignant', 'benign']


# drawing line (hyperplane) that is in the same distance between 2 points that are the farthest from opposite data sets
# we want to have distance to be as large as possible = more separate = more accurate predictions
# to draw hyperplane it is needed to change 2 dimension plane to 3D by using Kernel function
# soft or hard margin. Margin = space between hyperplane and first the closest point. Soft = can exist points in margin


# hard margin (C = 0) or soft margin, equal 2 = double the amount of points that are allowed to be
classifier = svm.SVC(kernel="linear", C=2)
classifier.fit(x_train, y_train)

y_prediction = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_prediction)  # why not -> accuracy = classifier.score(x_test, y_test)
print(accuracy)


# ############################# use KNN classifier to check what are the difference between these two methods:
classifier2 = KNeighborsClassifier(n_neighbors=8)
classifier2.fit(x_train, y_train)

y_prediction2 = classifier2.predict(x_test)
acc2 = accuracy_score(y_test, y_prediction2)
print(acc2)
