import time
import numpy as np
import matplotlib.pyplot as plt


def batch_td(training_episodes, w=None, lambda_=0.3, alpha=0.005, epsilon=1e-7):
    num_states = len(training_episodes[0][0])
    if w is None:
        w = np.zeros(num_states) * 0.5
    delta_w = w + 1
    start = time.time()
    while np.linalg.norm(delta_w) > epsilon:
        for episode in training_episodes:
            e = np.zeros(num_states)
            delta_w = np.zeros(num_states)
            x0 = episode[0]
            P0 = w.dot(x0)

            for n, x in enumerate(episode[1:]):
                terminal = n == len(episode) - 2
                e += x0

                if terminal:
                    P = x
                else:
                    P = w.dot(x)

                delta_w += alpha * (P - P0) * e

                if not terminal:
                    e *= lambda_
                    x0 = x.copy()
                    P0 = P

            w += delta_w

        if time.time() - start > 2:
            print 'delta_w norm:', np.linalg.norm(delta_w), w
            start = time.time()

    return w


def walk(num_states):
    def to_array(state):
        v = np.zeros(num_states)
        v[state] = 1
        return v

    state = int(np.median(range(num_states)))
    episode = [to_array(state)]
    while state not in (-1, num_states):
        state += np.random.choice((-1, 1))
        if state == -1:
            episode.append(0)
        elif state == num_states:
            episode.append(1)
        else:
            episode.append(to_array(state))
    return episode


def get_true_w(num_states):
    true_w = np.array([float(i) / num_states for i in xrange(num_states)])
    true_w[-1] = 0.0
    return true_w


def rmse(v1, v2):
    return np.sqrt(np.mean(np.square(v1 - v2)))


def fig3():
    true_w = get_true_w(7)
    training_sets = [[walk(7) for _ in xrange(10)] for _ in xrange(100)]
    lambda_range = np.arange(0.0, 1.1, 0.1)
    errors_by_lambda = []

    for lambda_ in lambda_range:
        print 'lambda:', lambda_
        errors = []
        for training_set in training_sets:
            w = batch_td(7, training_set, lambda_=lambda_)
            errors.append(rmse(w[1:6], true_w[1:6]))
        errors_by_lambda.append(np.mean(errors))

    build_plot(lambda_range, errors_by_lambda, 'lambda', 'RMSE')
    return lambda_range, errors_by_lambda


def build_plot(x, y, xlab, ylab):
    plt.ion()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def test():
    training_set = [walk(5) for _ in xrange(10)]
    return (
        batch_td(training_set, lambda_=0),
        batch_td(training_set, lambda_=0.3),
        batch_td(training_set, lambda_=0.6),
        batch_td(training_set, lambda_=1),
    )

