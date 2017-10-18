#
# USI - Universita della svizzera italiana
#
# Machine Learning - Fall 2017
#
# Assignment 1: Neural Networks
# Code Skeleton

import numpy as np
import matplotlib.pyplot as plt


## Part 1

def get_part1_data():
    """
	Returns the toy data for the first part.
	"""
    X = np.array([[1, 8], [6, 2], [3, 6], [4, 4], [3, 1], [1, 6],
                  [6, 10], [7, 7], [6, 11], [10, 5], [4, 11]])
    T = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]).reshape(-1, 1)
    return X, T


def MSE(prediction, target):
    """
	Computes the Mean Squared Error of a prediction and its target
	"""
    y = prediction
    t = target
    n = prediction.size

    ## Implement
    meanCost = (np.sum((t-y)**2))/2/n

    ## End
    return meanCost


def dMSE(prediction, target):
    """
	Computes the derivative of the Mean Squared Error function.
	"""
    y = prediction
    t = target
    n = prediction.size

    ## Implement

    error = (y - t)/n

    ## End
    return error


class Perceptron:
    """
	Keeps track of the variables of the Perceptron model. Can be used for predicting and to compute the gradients.
	"""

    def __init__(self):
        """
		The variables are stored inside a dictionary to make them easily accessible.
		"""
        self.var = {
            "W": np.array([[.8], [-.5]]),
            "b": 2
        }

    def forward(self, inputs):
        """
		Implements the forward pass of the perceptron model and returns the prediction y. We need to 
		store the current input for the backward function.
		"""
        x = self.x = inputs
        W = self.var['W']
        b = self.var['b']

        ## Implement
        net = np.dot(x, W)

        y = net+b

        ## End
        return y

    def backward(self, error):
        """
		Backpropagates through the model and computes the derivatives. The forward function must be 
		run beforehand for self.x to be defined. Returns the derivatives without applying them using
		a dictionary similar to self.var.
		"""
        x = self.x

        ## Implement
        dW = np.dot(x.T, error)
        db = sum(error)

        ## End
        updates = {"W": dW,
                   "b": db}
        return updates


def train_one_step(model, learning_rate, inputs, targets):
    """
	Uses the forward and backward function of a model to compute the error and updates the model 
	weights while overwriting model.var. Returns the cost.
	"""

    ## Implement

    predictions = model.forward(inputs)

    error = dMSE(predictions, targets)
    updates = model.backward(error)

    for varstr, grad in updates.items():
        model.var[varstr] = model.var[varstr] - (grad * learning_rate)

    return MSE(predictions, targets)
    ## End


def plot_data(X, T):
    """
	Plots the 2D data as a scatterplot
	"""
    plt.scatter(X[:, 0], X[:, 1], s=40, c=T[:, 0], cmap=plt.cm.Spectral)


def plot_boundary(model, X, targets, threshold=0.0):
    """
	Plots the data and the boundary lane which separates the input space into two classes.
	"""
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    y = model.forward(X_grid)
    plt.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
    plot_data(X, targets)
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])


def run_part1():
    """
	Train the perceptron according to the assignment.
	"""
    X, T = get_part1_data()
    p = Perceptron()
    for _i in range(15):
        error = train_one_step(p, 0.02, X, T)

    print("Part 1 error:", error)
    print("Part 1 accuracy:", compute_accuracy(p, X, T))
    plot_boundary(p, X, T)
    plt.show()



## Part 2
def twospirals(n_points=120, noise=1.6, twist=420):
    """
     Returns a two spirals dataset.
    """
    np.random.seed(0)
    n = np.sqrt(np.random.rand(n_points, 1)) * twist * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    X, T = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))
    T = np.reshape(T, (T.shape[0], 1))
    return X, T


def compute_accuracy(model, X, T):
    """
	Computes the average accuracy over this data.
	"""
    return np.mean(((model.forward(X) > 0.5) * 1 == T) * 1)


def sigmoid(x):
    """
	Implements the sigmoid activation function. 
	"""
    ## Implement
    x = 1/(1+np.exp(-x))

    ## End
    return x


def dsigmoid(x):
    """
	Implements the derivative of the sigmoid activation function. 
	"""
    ## Implement
    x = sigmoid(x) * (1 - sigmoid(x))

    ## End
    return x


def tanh(x):
    """
	Implements the hyperbolic tangent activation function.
	"""
    ## Implement
    x = np.tanh(x)

    ## End
    return x


def dtanh(x):
    """
	Implements the derivative of the hyperbolic tangent activation function. 
	"""
    ## Implement
    x = 1 - tanh(x)**2

    ## End
    return x


class NeuralNetwork:
    """
	Keeps track of the variables of the Multi Layer Perceptron model. Can be 
	used for predicting and to compute the gradients.
	"""

    def __init__(self):
        """
		The variables are stored inside a dictionary to make them easy accessible.
		"""
        ## Implement
        np.random.seed()

        self.var = {
            "W1": np.random.rand(2, 20),
            "b1": np.random.rand(1, 20),
            "W2": np.random.rand(20, 15),
            "b2": np.random.rand(1, 15),
            "W3": np.random.rand(15, 1),
            "b3": np.random.rand(1, 1)
        }

        self.net = {
            "net1": 0,
            "net2": 0,
            "net3": 0
        }

        self.a = {
            "a1": 0,
            "a2": 0,
            "a3": 0
        }

    ## End

    def forward(self, inputs):
        """
		Implements the forward pass of the MLP model and returns the prediction y. We need to
		store the current input for the backward function.
		"""
        a1 = self.x = inputs

        W1 = self.var['W1']
        b1 = self.var['b1']
        W2 = self.var['W2']
        b2 = self.var['b2']
        W3 = self.var['W3']
        b3 = self.var['b3']

        ## Implement
        net1 = a1.dot(W1) + b1
        a2 = tanh(net1)
        net2 = a2.dot(W2) + b2
        a3 = tanh(net2)
        net3 = a3.dot(W3) + b3
        y = sigmoid(net3)

        self.net["net1"] = net1
        self.net["net2"] = net2
        self.net["net3"] = net3

        self.a["a1"] = a1
        self.a["a2"] = a2
        self.a["a3"] = a3

        ## End
        return y

    def backward(self, error):
        """
		Backpropagates through the model and computes the derivatives. The forward function must be
		run before hand for self.x to be defined. Returns the derivatives without applying them using
		a dictionary similar to self.var.
		"""
        a1 = self.a['a1']
        W1 = self.var['W1']
        b1 = self.var['b1']
        net1 = self.net['net1']
        a2 = self.a['a2']
        W2 = self.var['W2']
        b2 = self.var['b2']
        net2 = self.net['net2']
        a3 = self.a['a3']
        W3 = self.var['W3']
        b3 = self.var['b3']
        net3 = self.net['net3']

        ## Implement
        dOUTPUT = error * dsigmoid(net3)
        d3 = (dOUTPUT.dot(W3.T)) * dtanh(net2)
        d2 = (d3.dot(W2.T)) * dtanh(net1)

        db3 = dOUTPUT.sum(axis = 0, keepdims = True)
        db2 = d3.sum(axis = 0, keepdims = True)
        db1 = d2.sum(axis = 0, keepdims = True)

        dW3 = a3.T.dot(dOUTPUT)
        dW2 = a2.T.dot(d3)
        dW1 = a1.T.dot(d2)

        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3}
        return updates


def gradient_check():
    """
	Computes the gradient numerically and analytically and compares them.
	"""
    X, T = twospirals(n_points=10)
    NN = NeuralNetwork()
    eps = 0.0001

    for key, value in NN.var.items():
        row = np.random.randint(0, NN.var[key].shape[0])
        col = np.random.randint(0, NN.var[key].shape[1])
        print("Checking ", key, " at ", row, ",", col)

        y = NN.forward(X)
        error = dMSE(y, T)
        updates = NN.backward(error)
        analytic_grad = updates[key][row][col]

        x1 = MSE(y, T)
        NN.var[key][row][col] += eps
        y = NN.forward(X)
        x2 = MSE(y, T)

        ## End
        numeric_grad = (x2 - x1) / eps
        print("numeric grad: ", numeric_grad)
        print("analytic grad: ", analytic_grad)
        if abs(numeric_grad - analytic_grad) < 0.00001:
            print("[OK]")
        else:
            print("[FAIL]")


def split_data(data, target):
    """
    Split the input data into two sets.
	"""
    Xcopy = np.copy(data)
    Xcopy = np.append(Xcopy, target, axis=1)

    np.random.seed()
    np.random.shuffle(Xcopy)

    XTrain = Xcopy[:int(Xcopy.shape[0] * 0.7), :2] #TODO capire in che proporzione splittare
    XTest = Xcopy[int(Xcopy.shape[0] * 0.7):, :2]
    TTrain = Xcopy[:int(Xcopy.shape[0] * 0.7), 2]
    TTest = Xcopy[int(Xcopy.shape[0] * 0.7):, 2]

    return XTrain, XTest, np.array([TTrain]).T, np.array([TTest]).T


def run_part2():
    """
	Train the multi layer perceptron according to the assignment.
	"""
    X, T = twospirals()
    XTrain, XTest, TTrain, TTest = split_data(X, T)

    #gradient_check()

    learning_rates = [0.1, 0.01, 0.02, 0.001, 0.0001] #define various learning rates to compare

    nn = NeuralNetwork()
    errorTrain = [] # various arrays to save data that will be plotted
    accuracyTrain = []
    errorTest = []
    accuracyTest = []

    for j in learning_rates:
        for i in range(2):
            errorTrain.append(train_one_step(nn, j, XTrain, TTrain))
            accuracyTrain.append(compute_accuracy(nn, XTrain, TTrain))

            predictions = nn.forward(XTest)
            errorTest.append(MSE(predictions, TTest))
            accuracyTest.append(compute_accuracy(nn, XTest, TTest))

            if i%1000 == 0 and False:
                plot_boundary(nn, XTrain, TTrain, threshold=0.5)
                plt.title("Train data\nEpoch: "+str(i)+"    Error: "+str(errorTrain[-1]))
                plt.show(block=False)
                plt.pause(0.001)
                plt.clf()

        plt.plot(errorTrain, label=str(j))
        errorTrain.clear()
        errorTest.clear()
        accuracyTrain.clear()
        errorTest.clear()

    plt.title("Learning rate comparison")
    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    plot_boundary(nn, XTrain, TTrain, threshold=0.5)
    plt.title("Train data\nEpochs: " + str(i+1) + "    Error: " + str(np.mean(errorTrain)))
    plt.show()

    plot_boundary(nn, XTest, TTest, threshold=0.5)
    plt.title("Test data    Error: " + str(np.mean(errorTest)))
    plt.show()

    plt.plot(errorTrain)
    plt.plot(errorTest)
    plt.title("Error in training: "+str(np.mean(errorTrain))+"\nError in testing: "+str(np.mean(errorTest)))
    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.show()

    plt.plot(accuracyTrain)
    plt.plot(accuracyTest)
    plt.title("Accuracy in training: " + str(np.mean(accuracyTrain)) + "\nAccuracy in testing: " + str(np.mean(accuracyTest)))
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.show()


## Part 3
class BetterNeuralNetwork:
    """
	Keeps track of the variables of the Multi Layer Perceptron model. Can be 
	used for prediction and to compute the gradients.
	"""

    def __init__(self):
        """
		The variables are stored inside a dictionary to make them easily accessible.
		"""
        ## Implement
        # W1_in = ...
        # W1_out = ...
        # W2_in = ...
        # W2_out = ...
        # W3_in = ...
        # W3_out = ...

        self.var = {
            # "W1": (...),
            # "b1": (...),
            # "W2": (...),
            # "b2": (...),
            # "W3": (...),
            # "b3": (...),
        }

    ## End

    def forward(self, inputs):
        """
		Implements the forward pass of the MLP model and returns the prediction y. We need to 
		store the current input for the backward function.
		"""
        x = self.x = inputs

        W1 = self.var['W1']
        b1 = self.var['b1']
        W2 = self.var['W2']
        b2 = self.var['b2']
        W3 = self.var['W3']
        b3 = self.var['b3']

        ## Implement



        ## End
        return y

    def backward(self, error):
        """
		Backpropagates through the model and computes the derivatives. The forward function must be 
		run before hand for self.x to be defined. Returns the derivatives without applying them using
		a dictonary similar to self.var.  
		"""
        x = self.x
        W1 = self.var['W1']
        b1 = self.var['b1']
        W2 = self.var['W2']
        b2 = self.var['b2']
        W3 = self.var['W3']
        b3 = self.var['b3']

        ## Implement



        ## End
        updates = {"W1": dW1,
                   "b1": db1,
                   "W2": dW2,
                   "b2": db2,
                   "W3": dW3,
                   "b3": db3}
        return updates


def competition_train_from_scratch(testX, testT):
    """
	Trains the BetterNeuralNet model from scratch using the twospirals data and calls the other 
	competition funciton to check the accuracy.
	"""
    trainX, trainT = twospirals(n_points=250, noise=0.6, twist=800)
    NN = BetterNeuralNetwork()

    ## Implement



    ## End

    print("Accuracy from scratch: ", compute_accuracy(NN, testX, testT))


def competition_load_weights_and_evaluate_X_and_T(testX, testT):
    """
	Loads the weight values from a file into the BetterNeuralNetwork class and computes the accuracy.
	"""
    NN = BetterNeuralNetwork()

    ## Implement



    ## End

    print("Accuracy from trained model: ", compute_accuracy(NN, testX, testT))


def main():

    #run_part1()
    run_part2()

if __name__ == "__main__":
    main()