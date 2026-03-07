import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x)
    
    if rng is not None:
        rand = rng.random(x.shape)
    else:
        rand = np.random.random(x.shape)

    # True represent keep
    keep_mask = rand < (1 - p)

    # Scaling factor
    scale = 1 / (1 - p)

    dropout_pattern = keep_mask.astype(int) * scale
    output = x * dropout_pattern

    return output, dropout_pattern