import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    seq_len = int(seq_len)
    d_model = int(d_model)

    pos = np.arange(seq_len, dtype=np.float64)[:, None]  # (seq_len, 1)

    # Even indices (0, 2, 4,...) (Use sin)
    even_idx = np.arange(0, d_model, 2, dtype=np.float64)   # (ceil(d_model/2),)

    # div_term: base^(2i/d_model), angles: pos/div_term
    div_term = base ** (even_idx / d_model)   # (ceil(d_model/2),)
    angles = pos / div_term[None, :]          # (seq_len, ceil(d_model/2))

    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)    # even columns (0,2,4,...) are sin
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])   # odd columns are cos

    return pe
    