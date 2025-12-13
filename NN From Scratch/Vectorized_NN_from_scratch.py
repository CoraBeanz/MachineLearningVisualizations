#everything here is done with numpy
import numpy as np

def vec_linreg_cost_function(w, X, y, b):
    m = X.shape(0)
    f = np.dot(w, X) + b
    error = f - y
    J_wb = (1 / (2 * m)) * np.sum(error ** 2)
    return J_wb

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def linear(z):
    return z

def vec_grad_descent(X, y, activation, w_init, b_init=0, iterations=1000, a=0.01):
    m = X.shape[0]
    g = activation

    # error handling
    if activation not in (linear, sigmoid):
        print("This function only supports linear and sigmoid at the current time.")
        return None, None
    if w_init is None:
        w = [0.0 for _ in range(m)]
    else:
        w = w_init

    for k in range(iterations):
        z = np.dot(X,w) + b
        error = g(z) - y
        dJdw = (1 / (m)) * np.dot(X.T,error)
        dJdb = (1 / (m)) * np.sum(error)
        w = w - a * dJdw
        b = b - a * dJdb
    return w, b

def vec_logreg_cost_function(w, X, y, b):
    m = X.shape(0)
    z = np.dot(X, w) + b
    f = sigmoid(z)
    loss = -np.dot(y, np.log(f)) + np.dot(1 - y, np.log(1 - f))
    J_wb = (1 / m) * loss
    return J_wb

def vec_dense(AT, W, b, activation):
    g = activation
    z = np.matmul(AT,W) + b
    a_out = g(z)
    return a_out

if __name__ == "__main__":
    #dummy variables for linear regression
    X1 = np.array([
        [0,1],
        [2,3],
        [4,5]
        ])
    y1 = np.array([6,7,8])

    #dummy variables for logistic regression
    X2 = np.array([
        [0,1],
        [2,3],
        [4,5]
        ])
    y2 = np.array([0,1,1])