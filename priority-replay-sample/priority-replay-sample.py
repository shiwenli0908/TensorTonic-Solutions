def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    # Powered priorities
    scaled_priorities = [p ** alpha for p in priorities]

    # Normalize to get probabilities
    total = sum(scaled_priorities)
    probabilities = [sp / total for sp in scaled_priorities]

    # Compute raw importance sampling weights
    N = len(probabilities)
    weights = [(N * prob) ** (-beta) for prob in probabilities]

    # Normalize weights by the maximum weight
    max_weight = max(weights)
    normalized_weights = [w / max_weight for w in weights]

    return [probabilities, normalized_weights]