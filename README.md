# MachineLearningLearning
Machine Learning programs used for deeper exploring the topic

Artificial Learning is a technique which enables machines to mimic human behavior.
Machine Learning is a subset of AI which uses statistical methods to enable machines to improve with experience.
Deep Learning is a particular part of ML that needs larger amount of data. DL is inspired by the functionality of out brain cells called neurons which led to the concept of Artificial Neuron Network. The ANN takes the data connection between all the artificial neurons and adjust the according to the data pattern. 

Neural Network is consisted of input layer, arbitrary amount of hidden layers, output layers, set of weights and biases (between each layer), and choice of activation function (for each hidden layer). 
Training of Neural Network means fine-tuning weights and biases from the input data, where the goal is to find the best set of weights and biases that minimizes a loss function. The loss function is a way to evaluate the “goodness” of our predictions (ex. sum of squares), so the function measures prediction. 
Each iteration of learning requires: feed forward (calculating the predicted output) and back propagation (updating the weights and biases). From random set of weights and biases variables, the neural network will find proper set by training, checking, testing and adjusting. Organize all the activations from one layer into a column as a vector. Organize weights as a matrix where each row corresponds to the connection between layers and neurons. Neurons that fire together wire together. 

Neuron is a function that result is a number between 0 – 1.
Activation function is a measures how the positive the relevant weighted sum is (result is between 0 - 1.)
Weight is the strength of the connection. They show how increasing the input influence on the output.
Biases tell us how the high the weighted sum needs to be before the neuron starts getting meaningful active. Biases show how far off our predictions are from real values.

Supervised learning is a method in which the machine use labeled data. We distinguish regression method (predicting continuous quantity for limitless possibilities as weights for example) and classification (predicting a label or class). Supervised ML maps input to output, so the training data acts like a guide.
Unsupervised learning is a method where the machine is trained on unlabelled data without any guidance to shows group by patterns as association (similarity, trends, and frequently matches) and clustering (group by similarity as K means).
Reinforcement learning is a method popular used in robotics, where an agent as robot interacts with its environment by producing actions and discover the environment. By those actions the agent gets punishments or rewards and learns from this experience. Inputs depend on the actions.

7 steps of the Machine Learning:
1.	Gathering, collect data (features, labels)
2.	Prepare data (training and test data, randomize order, divide data type 50/50)
3.	Choosing a model
4.	Train the data (weights, biases)
5.	Evaluate model by accuracy and loss function
6.	Parameter tuning to increase accuracy
7.	Prediction with testing data

•	y = a*x + b => output = weight(slope) * input + bias(y-intercept)
•	Training data -> Model -> Predictions -> Test and Update weights and bias
•	Split data to training data 80%, test 20% (example that shows pattern)

