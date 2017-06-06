import time
import numpy as np
import matplotlib.pyplot as plt


def batch_td(num_states, training_episodes, w=None, lambda_=0.3, alpha=0.005, epsilon=1e-5):
    if w is None:
        w = np.zeros(num_states) * 0.5
    
    w_gradient  = w + 1
    start = time.time()
    while np.linalg.norm(w_gradient) > epsilon:
        for episode in training_episodes:
            e = np.zeros(num_states)
            w_gradient = np.zeros(num_states)
            s0 = episode[0]
            r = 0
            for s in episode:
                if s == num_states - 1:
                    r = 1
                w_gradient = alpha * (r + w[s] - w[s0]) * e
                e[s0] += 1
                e *= lambda_
                s0 = s
            w += w_gradient

        if time.time() - start > 2:
            print 'w gradient norm:', np.linalg.norm(w_gradient)
            start = time.time()

    return w


def walk(num_states):
    state = int(np.median(range(num_states)))
    episode = [state]
    while state not in (0, num_states - 1):
        state += np.random.choice((-1, 1))
        episode.append(state)
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


def test(lambda_=0.3):
    training_set = [walk(7) for _ in xrange(10)]
    return batch_td(7, training_set, lambda_=lambda_)

