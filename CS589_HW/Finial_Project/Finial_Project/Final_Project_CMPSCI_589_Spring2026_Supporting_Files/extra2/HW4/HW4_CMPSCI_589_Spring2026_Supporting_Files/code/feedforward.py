import numpy as np
# import pandas as pd


def xaiver_initialization(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))


class Feedforward:
    def __init__(self, layer_size: list[int]):
        self.weights = []
        self.bias = []
        for i in range(len(layer_size) - 1):
            self.weights.append(xaiver_initialization(layer_size[i], layer_size[i + 1]))
            self.bias.append(np.zeros((1, layer_size[i + 1])))

    # activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.bias)):
            z = np.dot(activations[-1], w) + b
            zs.append(z)
            if i == len(self.weights) - 1:
                A = self.softmax(z)
            else:
                A = self.sigmoid(z)
            activations.append(A)
        return A, zs, activations

    def compute_loss(self, Y_pred, Y_true, lambda_=0.0):
        m = Y_true.shape[0]
        eps = 1e-15
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        loss = -(1 / m) * np.sum(Y_true * np.log(Y_pred))
        reg = (lambda_ / (2 * m)) * sum(np.sum(w**2) for w in self.weights)
        return loss + reg

    # 跟ppt的公式一样
    def backward(self, Y_true, Y_pred, activations, lambda_=0.0):
        m = Y_true.shape[0]

        dWs = [None] * len(self.weights)
        dbs = [None] * len(self.bias)

        # 输出层 delta
        dZ = Y_pred - Y_true

        # 从后往前
        for l in reversed(range(len(self.weights))):
            A_prev = activations[l]

            dW = (1 / m) * np.dot(A_prev.T, dZ)
            db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

            # regularization: 只对 weight，不对 bias
            dW += (lambda_ / m) * self.weights[l]

            dWs[l] = dW
            dbs[l] = db

            if l > 0:
                dA_prev = np.dot(dZ, self.weights[l].T)
                dZ = dA_prev * (activations[l] * (1 - activations[l]))
        return dWs, dbs

    def step(self, dWs, dbs, learning_rate):
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * dWs[l]
            self.bias[l] -= learning_rate * dbs[l]

