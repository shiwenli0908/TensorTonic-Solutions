import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    x = np.asarray(w)
    g = np.asarray(g)
    G = np.asarray(G)

    # Step 1: Accumulate Squared Gradients
    new_G = G + g ** 2

    # Step 2: Parameter update
    new_w = w - lr * g / np.sqrt(new_G + eps)

    return new_w, new_G
    