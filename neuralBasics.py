import numpy as np
import matplotlib.pyplot as mpl

input_size = 2
hidden_size = 3
output_size = 1

# initialize weights and biases at the start
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01  # Weight matrix for the hidden layer
    b1 = np.zeros((hidden_size, 1))                       # Bias for the hidden layer
    W2 = np.random.randn(output_size, hidden_size) * 0.01 # Weight matrix for the output layer
    b2 = np.zeros((output_size, 1))                       # Bias for the output layer
    
    return W1, b1, W2, b2

# define an activation function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# calculating the output
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1         # Linear step for hidden layer
    A1 = sigmoid(Z1)                # Activation step for hidden layer
    Z2 = np.dot(W2, A1) + b2        # Linear step for output layer
    A2 = sigmoid(Z2)                # Activation step for output layer
    
    return A1, A2


def compute_cost(A2, Y):
    m = Y.shape[1]  # Number of examples
    cost = np.sum((A2 - Y) ** 2) / (2 * m)
    return cost


def backward_propagation(X, Y, A1, A2, W2):
    m = X.shape[1]  # Number of examples
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * A1 * (1 - A1)  # Derivative of sigmoid
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2


def train(X, Y, input_size, hidden_size, output_size, iterations=1000, learning_rate=0.01):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    costs = []
    
    for i in range(iterations):
        A1, A2 = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y)
        costs.append(cost)
        
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")
    
    return W1, b1, W2, b2, costs


X = np.random.randn(2, 100)  # Example input data (2 features, 100 examples)
Y = (np.sum(X, axis=0) > 0).reshape(1, 100)  # Example output data (binary classification)

W1, b1, W2, b2, costs = train(X, Y, input_size, hidden_size, output_size)

mpl.plot(costs)
mpl.xlabel("Iterations")
mpl.ylabel("Cost")
mpl.title("Training Cost")
mpl.show()

