import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)

    if r1.shape != r2.shape:
        raise ValueError("Both raters must have the same number of samples")

    n = len(r1)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = np.mean(r1 == r2)

    # Expected agreement
    labels = np.union1d(r1, r2)

    p_e = 0.0
    for label in labels:
        p1 = np.sum(r1 == label) / n
        p2 = np.sum(r2 == label) / n
        p_e += p1 * p2

    # Degenerated case: p_e = 1.0
    if np.isclose(p_e, 1.0):
        return 1.0 if np.isclose(p_o, 1.0) else 0.0

    # Kappa formula
    kappa = (p_o - p_e) / (1 - p_e)

    return float(kappa)

    