def get_value(N, B, bankroll=0, depth=1):
    prob_each = 1.0 / N
    prob_bad = prob_each * sum(B)
    reward = prob_each * sum([i + 1 for i in range(N) if not B[i]])
    expected_value = reward - bankroll * prob_bad

    if expected_value > 0:
        expected_value += max([
            get_value(N, B, bankroll + i + 1, depth + 1) for
            i in xrange(N) if not B[i]
        ] + [0]) / (N ** depth)
        print bankroll, expected_value
    return expected_value
