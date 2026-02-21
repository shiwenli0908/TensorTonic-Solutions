import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(int(steps)):
        # Forward
        z = X @ w + b      # (n,)
        p = _sigmoid(z)    # (n,) predicted P(y=1|x)

        # Gradient (average over samples)
        error = p - y           # (n,)
        dw = X.T @ error / n    # (d,)
        db = np.sum(error) / n  # scalar

        # Update
        w -= lr * dw
        b -= lr * db

    return w, b

        

        

        
    