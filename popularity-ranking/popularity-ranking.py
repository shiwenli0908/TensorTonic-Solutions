def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    result = []

    for avg_rating, vote_count in items:
        weighted_rating = (
            (vote_count / (vote_count + min_votes)) * avg_rating 
            + (min_votes / (vote_count + min_votes)) * global_mean
        )

        result.append(weighted_rating)

    return result