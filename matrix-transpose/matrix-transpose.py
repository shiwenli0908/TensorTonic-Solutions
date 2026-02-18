import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    rows, cols = A.shape

    result = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            result[j, i] = A[i, j]

    #result = np.transpose(A)

    return result
