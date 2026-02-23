import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Convert to ndarray
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    # Update runing average of squared gradients
    new_s = beta * s + (1 - beta) * g * g

    # Parameter update (adaptive learning rate)
    new_w = w - lr / np.sqrt(new_s + eps) * g

    return new_w, new_s
    