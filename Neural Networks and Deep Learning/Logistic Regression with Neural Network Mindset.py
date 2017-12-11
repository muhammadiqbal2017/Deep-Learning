import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# Finding the values

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

# Reshaping the training and test data sets

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardizing the dataset

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# Activation Function (Sigmoid)

def sigmoid(z):

	s = 1 / (1 + np.exp(-z))

	return s

print(sigmoid(0))

# Initializing Parameters

def initializing(dim):
	
	w = np.zeros(shape=(dim, 1))
	b = 0

	return w, b

# Forward and Backward Propagation

def propagate(w, b, X, Y):

	m = X.shape[1]

	#Forward Propagation
	A = sigmoid(np.dot(w.T, X) + b)
	cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

	#Backward Propagation
	dw = (1/m) * np.dot(X, (A-Y).T)
	db = (1/m) * np.sum(A - Y)

	grads = {"dw":dw,
			"db":db}

	return grads, cost

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print("dw: ", grads["dw"])
print("db: ", grads["db"])
print("cost: ", cost)

# Optimization function

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost= False):

	costs = []

	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)

		dw = grads["dw"]
		db = grads["db"]

		w = w - learning_rate * dw
		b = b - learning_rate * db

		if i % 100 == 0:
			costs.append(cost)

		if i % 100 == 0 and print_cost:
			print("Cost after iteration %i: %f" % (i, cost))

	params = {"w": w,
			"b": b}

	grads = {"dw": dw,
			"db": db}

	return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

# Computing predictions

def predict(w, b, X):

	m = X.shape[1]
	y_prediction = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X) + b)

	for i in range (A.shape[1]):

		y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

	return y_prediction

print("predictions = " + str(predict(w, b, X)))

# Overall model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

	w, b = initializing(X_train.shape[0])

	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
	
	w = parameters["w"]
	b = parameters["b"]

	y_prediction_test = predict(w, b, X_test)
	y_prediction_train = predict(w, b, X_train)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - Y_test)) * 100))

	d = {"costs": costs,
	         "Y_prediction_test": y_prediction_test, 
	         "Y_prediction_train" : y_prediction_train, 
	         "w" : w, 
	         "b" : b,
	         "learning_rate" : learning_rate,
	         "num_iterations": num_iterations}
	    
	return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1000, learning_rate = 0.005, print_cost = True)

# Test

my_image = "3.jpg" #Your image here

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")