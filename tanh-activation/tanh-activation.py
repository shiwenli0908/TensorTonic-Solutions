import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x_arr = np.asarray(x)

    exp_pos = np.exp(x_arr)
    exp_neg = np.exp(-x_arr)

    return (exp_pos - exp_neg) / (exp_pos + exp_neg)