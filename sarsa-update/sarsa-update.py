def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    # Deep copy
    new_q_table = [row[:] for row in q_table]

    # Compute TD error
    td = reward + gamma * q_table[next_state][next_action] - q_table[state][action]

    # Update the Q-value for the current state-action
    new_q_table[state][action] = q_table[state][action] + alpha * td

    return new_q_table
    