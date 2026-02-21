import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)

    # Determine target length L
    if max_len is not None:
        L = max_len
    else:
        L = max((len(seq) for seq in seqs), default=0)

    # Initialize result array with pad_value
    result = np.full((N, L), pad_value, dtype=np.int64)

    # Fill with actual sequence values
    for i, seq in enumerate(seqs):
        trunc = seq[:L]    # truncate if longer than L
        result[i, :len(trunc)] = trunc

    return result