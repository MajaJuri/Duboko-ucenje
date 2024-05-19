import numpy as np
import matplotlib.pyplot as plt
import data


def relu(s):
    return np.maximum(0, s)


def relu_derivative(s):
    return np.where(s > 0, 1, 0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    probs = exp_z / np.sum(exp_z)
    return probs


def train(X, y_true, hidden_size, param_niter=int(10000), learning_rate=0.05, param_lambda=0.001):
    input_size = X.shape[1]
    output_size = max(y_true) + 1
    N = X.shape[0]

    W1 = np.random.normal(0, 1, size=(input_size, hidden_size))
    #b1 = np.random.normal(0, 1, size=(1, hidden_size))
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.normal(0, 1, size=(hidden_size, output_size))
    #b2 = np.random.normal(0, 1, size=(1, output_size))
    b2 = np.zeros((1, output_size))

    Yoh = data.class_to_onehot(y_true)
    #print(self.W1, self.W2, self.b1, self.b2)
    for i in range(param_niter):
        # forward pass
        s1 = np.dot(X, W1) + b1
        h1 = relu(s1)
        s2 = np.dot(h1, W2) + b2
        probs = []
        for s in s2:
            probs.append(softmax(s))
        probs = np.array(probs)

        # gubitak
        loss = - np.sum(Yoh * np.log(probs))/N + param_lambda * (np.linalg.norm(W1) + np.linalg.norm(W2))

        if i % 100 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dLi_ds2 = probs - Yoh
        dW2 = np.dot(dLi_ds2.T, h1).T + param_lambda * W2
        db2 = np.dot(np.ones(N), probs - Yoh) + param_lambda * b2
        dLi_dh1 = np.dot(dLi_ds2, W2.T)
        dh1_ds1 = relu_derivative(s1)
        dLi_ds1 = dLi_dh1 * dh1_ds1 # N x H
        dW1 = np.dot(X.T, dLi_ds1) + param_lambda * W1
        db1 = np.dot(np.ones(N), dLi_ds1 + param_lambda * b1)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        #print(self.W1, self.b1, self.W2, self.b2)
    return W1, b1, W2, b2


def classify(X, W1, b1, W2, b2):
    s1 = np.dot(X, W1) + b1
    h1 = relu(s1)
    s2 = np.dot(h1, W2) + b2
    probs = []
    for s in s2:
        probs.append(softmax(s))
    probs = np.array(probs)
    return probs


def decision_function(W1, b1, W2, b2):
    def classify_nested(X):
        X = (X - np.mean(X)) / np.std(X)
        probs = classify(X, W1, b1, W2, b2)
        y_pred = np.argmax(probs, axis=1) # ili y_pred = np.max(probs, axis=1)
        return y_pred
    return classify_nested


if __name__ == "__main__":
    np.random.seed(100)

    K = 6 # broj komponenata, zadano
    C = 2 # broj klasa, zadano
    N = 20 # broj podataka u klasi, proizvoljno

    # stvaranje podataka i standardizacija
    X, Y_true = data.sample_gmm_2d(K=K, C=C, N=N)
    X = (X - np.mean(X)) / np.std(X)

    # hiperparametri
    hidden_size = 5
    param_niter = int(10000)
    learning_rate = 0.05
    param_lambda = 0.001

    # treniranje modela
    W1, b1, W2, b2 = train(X, Y_true, hidden_size, param_niter=param_niter, learning_rate=learning_rate, param_lambda=param_lambda)
    #print(W1, W2, b1, b2)

    # evaluacija modela
    probs = classify(X, W1, b1, W2, b2)
    Y_pred = np.argmax(probs, axis=1)
    #print(Y_pred)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decision_function(W1, b1, W2, b2), rect)
    data.graph_data(X, Y_true, Y_pred)
    plt.show()

