import numpy as np

# Activation function and its derivative
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

# Neural Network with one hidden layer
class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.rand(2, 2)  # Weights from input to hidden
        self.w2 = np.random.rand(2, 1)  # Weights from hidden to output
        self.b1 = np.random.rand(1, 2)  # Bias for hidden layer
        self.b2 = np.random.rand(1, 1)  # Bias for output layer

    def feedforward(self, X):
        self.hidden = sigmoid(np.dot(X, self.w1) + self.b1)
        self.output = sigmoid(np.dot(self.hidden, self.w2) + self.b2)
        return self.output

    def backpropagate(self, X, y):
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)
        
        hidden_error = output_delta.dot(self.w2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)
        
        self.w2 += self.hidden.T.dot(output_delta) * 0.1
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * 0.1
        self.w1 += X.T.dot(hidden_delta) * 0.1
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * 0.1

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, y)

# Data (XOR function)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train and test the network
nn = NeuralNetwork()
nn.train(X, y)
print("Predictions after training:")
print(np.round(nn.feedforward(X)))
