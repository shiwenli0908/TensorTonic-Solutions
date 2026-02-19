def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = float(x0) 
    a = float(a)
    b = float(b)
    c = float(c)    # c do not influence the gradient

    for _ in range(int(steps)):
        grad = 2 * a * x + b
        x = x - lr * grad

    return x