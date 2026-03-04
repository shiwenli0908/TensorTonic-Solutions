import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("a and b must be 1D array of equal length!")

    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0.0 or nb == 0.0:
        return 0.0

    return float(dot / (na * nb))