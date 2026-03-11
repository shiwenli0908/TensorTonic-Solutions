import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.asarray(v)

    if v.ndim == 1:
        return float(np.sqrt(np.sum(v ** 2)))
    elif v.ndim == 2:
        return np.sqrt(np.sum(v ** 2, axis=1))