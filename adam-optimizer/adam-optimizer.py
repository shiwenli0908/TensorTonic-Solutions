import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Convert to numpy array (accept scalars or arrays)
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    t = int(t)    # Ensure integer
    
    # Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * grad

    # Update biased second row moment estimate
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias corrections (t is 1-based)
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # Parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new