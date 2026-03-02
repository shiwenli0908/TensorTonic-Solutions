def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    new_values = []

    # Outer loop over states s
    for s in range(num_states):
        best = float("-inf")

        num_actions = len(transitions[s])

        # Middle loop over qctions q
        for a in range(num_actions):
            exp_next = 0.0    # Compute sum_{s'} T(s,a,s') * V(s')

            # Inner loop over next states s_next
            for s_next in range(num_states):
                prob = transitions[s][a][s_next]
                exp_next += prob * values[s_next]

            # Compute Q(s,a)
            q = rewards[s][a] + gamma * exp_next

            # Trace max a Q(s,a)
            if q > best:
                best = q

        new_values.append(best)

    return new_values

            

            
                