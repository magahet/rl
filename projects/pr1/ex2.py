import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def batch_td(training_episodes, w=None, lambda_=0.3, alpha=0.025, epsilon=1e-7):
    num_states = len(training_episodes[0][0])
    if w is None:
        w = np.ones(num_states) * 0.5
    delta_w = w + 1
    start = timer()
    while np.linalg.norm(delta_w) > epsilon:
    # for i in xrange(5):
        delta_w = np.zeros(num_states)
        for episode in training_episodes:
            e = np.zeros(num_states)
            x0 = episode[0]
            P0 = w.dot(x0)

            for n, x in enumerate(episode[1:]):
                terminal = n == len(episode) - 2
                e += x0

                if terminal:
                    P = x
                else:
                    P = w.dot(x)

                # print 'dw', n, alpha * (P - P0) * e, 'e', e, 'P', P, 'P0', P0
                delta_w += alpha * (P - P0) * e

                if not terminal:
                    e *= lambda_
                    x0 = x.copy()
                    P0 = P

        w += delta_w

        if timer() - start > 2:
            print 'delta_w norm:', np.linalg.norm(delta_w), w
            start = timer()

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
    true_w = np.array([float(i) / (num_states + 1) for i in xrange(1, num_states + 1)])
    return true_w


def rmse(v1):
    v2 = get_true_w(len(v1))
    return np.sqrt(np.mean(np.square(v1 - v2)))


def find_alpha():
    training_set = [walk(5) for _ in xrange(10)]
    for alpha in np.arange(0.001, 0.101, 0.001):
        start = timer()
        e = rmse(batch_td(training_set, alpha=alpha, lambda_=0))
        end = timer()
        print alpha, e, end - start


def fig3():
    true_w = get_true_w(5)
    training_sets = [[walk(5) for _ in xrange(10)] for _ in xrange(100)]
    lambda_range = np.arange(0.0, 1.1, 0.1)
    errors_by_lambda = []

    for lambda_ in lambda_range:
        errors = []
        for training_set in training_sets:
            w = batch_td(training_set, lambda_=lambda_)
            errors.append(rmse(w))
        errors_by_lambda.append(np.mean(errors))
        print 'lambda:', lambda_, 'rmse:', errors_by_lambda[-1]

    build_plot(lambda_range, errors_by_lambda, 'lambda', 'RMSE')
    return lambda_range, errors_by_lambda


def build_plot(x, y, xlab, ylab):
    plt.ion()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def simple():
    training_set = [
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 1]),
            1
        ]
    ]
    return batch_td(training_set, alpha=0.1, lambda_=0.1)

def more():
    training_set = [
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 1]),
            1
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 1]),
            1
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 1]),
            1
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 1]),
            1
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 1]),
            1
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 1, 0, 0, 0]),
            np.array([1, 0, 0, 0, 0]),
            0
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 1, 0, 0, 0]),
            np.array([1, 0, 0, 0, 0]),
            0
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 1, 0, 0, 0]),
            np.array([1, 0, 0, 0, 0]),
            0
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 1, 0, 0, 0]),
            np.array([1, 0, 0, 0, 0]),
            0
        ],
        [
            np.array([0, 0, 1, 0, 0]),
            np.array([0, 1, 0, 0, 0]),
            np.array([1, 0, 0, 0, 0]),
            0
        ],
    ]
    return batch_td(training_set, alpha=0.1, lambda_=0.1)

def test():
    training_set = [walk(5) for _ in xrange(10)]
    return (
        rmse(batch_td(training_set, lambda_=0)),
        rmse(batch_td(training_set, lambda_=0.3)),
        rmse(batch_td(training_set, lambda_=0.6)),
        rmse(batch_td(training_set, lambda_=1)),
    )

