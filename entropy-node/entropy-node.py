import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)

    # Empty node
    if y.size == 0:
        return 0.0

    # Count class occurences
    _, counts = np.unique(y, return_counts=True)

    # Calculate probabilities
    p = counts / counts.sum()

    # Stable logarithms: ignore zero-prob terms
    p_nonzero = p[p > 0]

    # Entropy in bits
    H = -sum(p_nonzero * np.log2(p_nonzero))

    return float(H)
    