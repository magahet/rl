def get_value(N, B, bankroll=0, history=()):
    history += (bankroll,)
    prob_each = 1.0 / N
    prob_bad = prob_each * sum(B)
    reward = prob_each * sum([i + 1 for i in range(N) if not B[i]])
    expected_value = reward - bankroll * prob_bad

    if expected_value > 0:
        future_values = [0] + [
            get_value(N, B, bankroll + i + 1, history) for
            i in xrange(N) if not B[i]
        ]
        future_value = sum([v for v in future_values if v >= 0])
        future_value /= float(N)
        print history, expected_value, future_value
        expected_value +=  future_value
    return expected_value
