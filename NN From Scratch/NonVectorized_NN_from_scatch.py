# No imports allowed besides math
import math

"""
Functions for linear regression

"""
def nonvec_linreg_cost_function(w, X, y, b):
    m = len(X)
    error = 0
    J_wb = 0
    for i in range(m):
        wX = 0
        for j in range(len(X[0])):
            wX += w[j] * X[i][j]
        f_wx = wX + b
        error += (f_wx - y[i]) ** 2
    J_wb += (1 / (2 * m)) * error
    return J_wb

def nonvec_logreg_cost_function(X, y, w, b):
    m = len(X)
    n = len(X[0])
    J_wb = 0
    for i in range(m):
        wX = 0
        loss = 0
        for j in range(n):
            wX += w[j] * X[i][j]
        z = wX + b
        loss = -y[i] * math.log(sigmoid(z)) - (1 - y[i]) * math.log(1 - sigmoid(z))
        J_wb += (1/m) * loss
    return J_wb

def sigmoid(z):
    g = 1 / (1 + math.exp(-z))
    return g

def linear(z):
    return z

def nonvec_general_grad_descent(X, y, activation, w_init, b_init=0, iterations=1000, a=0.01):
    
    # error handling
    if activation not in (linear, sigmoid):
        print("This function only supports linear and sigmoid at the current time.")
        return None, None

    m = len(X)
    n = len(X[0])
    b = b_init
    w = w_init
    g = activation

    # If w_init wasn't given, initialize to zeros
    if w_init is None:
        w = [0.0 for _ in range(n)]
    else:
        w = w_init

    for k in range(iterations):
        #initialize array variables/arrays with value 0
        dJdw = [0 for _ in range(n)]
        dJdb = 0
        error = [0 for _ in range(m)]
        for i in range(m):
            wX = 0
            for j in range(n):
                wX += w[j] * X[i][j]
            z = wX + b
            error[i] = g(z) - y[i]
            dJdb += (1/m) * error[i]
            for j in range(n):
                dJdw[j] += (1/m) * error[i] * X[i][j]
        for j in range(n):
            w[j] = w[j] - a * dJdw[j]
        b = b - a * dJdb
    return w, b

# for a 1d input 
def nonvec_dense_layer(a_in, W, b, layers):
    m = len(a_in)
    a_out = []

    # Error handling for W and b
    if W == None:
        W = [[1 for _ in range(m)] for _ in range(layers)]
    if b == None:
        b = [0.0 for _ in range(layers)]

    for i in range(layers):
        wX = 0
        z = 0
        for j in range(m):
            wX += a_in[j] * W[i][j]
        z = wX + b[i]
        a_out[i].append(sigmoid(z))
    return a_out

def sequential():
    return





if __name__ == "__main__":
    #dummy variables for linear regression
    X1 = [
        [0,1],
        [2,3],
        [4,5]
        ]
    y1 = [6,7,8]
    #dummy variables for logistic regression
    X2 = [
        [0,1],
        [2,3],
        [4,5]
        ]
    y2 = [0,1,1]



