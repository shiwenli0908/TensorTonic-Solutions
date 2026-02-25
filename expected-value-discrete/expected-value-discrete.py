import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Convert to numpy array
    x = np.asarray(x)
    p = np.asarray(p)

    # Check shape match
    if x.shape != p.shape:
        raise ValueError("x and p must have same shape")

    # Check probabilities sum to 1
    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1.0")

    return float(np.sum(x * p))
