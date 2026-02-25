def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Take top-k recommendations
    top_k = recommended[:k]

    # Convert relevent to set for fast lookup
    relevant_set = set(relevant)

    # Count hits
    hits = sum(1 for item in top_k if item in relevant_set)

    # Precision@k
    precision = hits / k

    # Recall@k
    recall = hits / len(relevant_set)

    return [float(precision), float(recall)]
    