import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """

    preds = np.array(predictions)

    T, N = preds.shape
    result = []

    for i in range(N):
        votes = preds[:, i]    # Votes of all trees for sample i
        count = np.bincount(votes)   # Count the votes of all classes
        majority = np.argmax(count)  # Class with max vote
        result.append(int(majority))
    
    return result    
    