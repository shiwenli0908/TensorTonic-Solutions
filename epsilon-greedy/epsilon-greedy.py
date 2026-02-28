import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    q = np.asarray(q_values)

    if q.ndim != 1 or q.size == 0:
        raise ValueError("q_values must be a non-empty 1D array")

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0,1]")

    n_actions = q.size

    # Greedy action
    greedy_action = int(np.argmax(q))

    # Choose RNG
    if rng is not None:
        rand = rng.random()
    else:
        rand = np.random.random()

    # epsilon-greedy decision
    if rand < epsilon:
        # random action
        if rng is not None:
            action = rng.integers(n_actions)
        else:
            action = np.random.randint(n_actions)

    else:
        action = greedy_action

    return int(action)
