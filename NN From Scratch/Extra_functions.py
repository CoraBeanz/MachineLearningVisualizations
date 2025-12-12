import math

def nonvec_linreg_grad_descent(X, y, w_init, b_init, iterations=1000, a=0.01):
    m = len(X)
    n = len(X[0])
    b = b_init
    w = w_init

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
            error[i] = wX + b - y[i]
            dJdb += (1/m) * error[i]
            for j in range(n):
                dJdw[j] += (1/m) * error[i] * X[i][j]
        for j in range(n):
            w[j] = w[j] - a * dJdw[j]
        b = b - a * dJdb
    return w, b

def sigmoid(z):
    g = 1 / (1 + math.exp(-z))
    return g

def nonvec_logreg_grad_descent(X, y, w_init, b_init=0, iterations=1000, a=0.01):
    m = len(X)
    n = len(X[0])
    w = w_init
    b = b_init

    # If w_init wasn't given, initialize to zeros
    if w_init is None:
        w = [0.0 for _ in range(n)]
    else:
        w = w_init

    for k in range(iterations):
        dJdw = [0 for _ in range(n)]
        dJdb = 0
        for i in range(m):
            wX = 0
            for j in range(n):
                wX += w[j] * X[i][j]
            z = wX + b
            error = sigmoid(z) - y[i]
            dJdb += (1/m) * error
            for j in range(n):
                dJdw[j] += (1/m) * error * X[i][j]
        for j in range(n):
            w[j] = w[j] - a * dJdw[j]
        b = b - a * dJdb
    return w, b