# Savaliya, Kuldip
# 1001_832_000
# 2022_10_10
# Assignment_02_01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        self.initialize_weights()



    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
          np.random.seed(seed)
          weights = np.random.randn(self.number_of_nodes,self.input_dimensions)
        else:
          weights = np.random.randn(self.number_of_nodes,self.input_dimensions)

        self.weights = weights
    		
		

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
		 
        self.weights = W 
        if self.weights.shape != (self.number_of_nodes, self.input_dimensions):
          return -1
        else:
            return None


    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights


    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        def hardlim_activation(net):
            hard_limit_list = []
            for i in range(len(net)):
                for j in range(len(net[0])):
                    hard_limit_list.append(1) if net[i][j] >= 0 else hard_limit_list.append(0)
            return hard_limit_list

        weights = self.weights
        no_of_nodes = self.number_of_nodes
        transfer_function = self.transfer_function
        net_val = np.dot(self.weights, X)
        if transfer_function.lower() == "hard_limit":
          act_func = hardlim_activation(net_val)
          act_func = np.reshape(act_func, (self.number_of_nodes, len(X[0])))
          return act_func
        else:
          return net_val




    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        X_plus = np.linalg.pinv(X)
        self.weights = np.dot(y, X_plus)
		
		
		
		

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        weights = self.weights

        for i in range(num_epochs):
            for j in range(0, X.shape[1], batch_size):
                batch = j + batch_size
                if batch > X.shape[1]:
                    batch = X.shape[1]
                predict_x = self.predict(X[:, j:batch])
                if learning.lower() == "filtered":
                  weights = (1-gamma) * weights + alpha * (np.dot((y[:, j:batch]), (X[:, j:batch]).T))
                elif learning.lower() == "delta":
                  weights += alpha * (np.dot((y[:, j:batch] - predict_x), (X[:, j:batch]).T))
                else:
                    weights += alpha * (np.dot((predict_x), (X[:, j:batch]).T))
                self.weights = weights



    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        mean_squared_error = np.square(y - self.predict(X)).mean()
        return mean_squared_error