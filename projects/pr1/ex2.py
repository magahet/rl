'''Reproduces the results of Sutton's 1988 paper.
GID: mmendiola3
'''


import numpy as np
import matplotlib.pyplot as plt


def batch_td(training_episodes, lambda_=0.3, alpha=0.025, epsilon=1e-7, batch=False):
    '''Implements the TD lambda algorithm as outlined in Sutton 1988.
    Can be run in batch or sequential mode.'''

    # Initialize weights, delta w, etc.
    num_states = len(training_episodes[0][0])
    w = np.ones(num_states) * 0.5
    delta_w = w + 1

    # Continue until convergence
    # This will break after the first run in sequential mode
    while np.linalg.norm(delta_w) > epsilon:
        delta_w = np.zeros(num_states)

        # Iterate through each state sequence
        for episode in training_episodes:

            # Reset delta_w when in sequential mode
            if not batch:
                delta_w = np.zeros(num_states)

            # Reset eligibility vector, initialize state and P
            e = np.zeros(num_states)
            x0 = episode[0]
            P0 = w.dot(x0)

            # Iterate through each state change
            for n, x in enumerate(episode[1:]):
                # Determine if state is terminal
                terminal = n == len(episode) - 2

                # Add 1 in the eligibility vector at the last state index
                e += x0

                # Calculate P, or use the final value 0 or 1
                if terminal:
                    P = x
                else:
                    P = w.dot(x)

                # Add to delta_w with Sutton's update equation
                delta_w += alpha * (P - P0) * e

                # Decay eligibility vector by lambda and increment state and P
                if not terminal:
                    e *= lambda_
                    x0 = x.copy()
                    P0 = P

            # Sequential mode weight update
            if not batch:
                w += delta_w

        # Batch mode weight update
        if batch:
            w += delta_w
        else:
            # Only run once through training set for sequential mode
            break

    return w


def walk(num_states):
    '''Generates a random walk ending with a 0 or 1 value.'''
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
    '''Generate true probabilities for each state in a random walk.
    This is [1/6, 2/6, 3/6, 4/6, 5/6] in the case of num_states=5
    '''
    true_w = np.array([float(i) / (num_states + 1) for i in xrange(1, num_states + 1)])
    return true_w


def rmse(v1):
    '''Calculate RMSE for a given predicted weight vector.'''
    v2 = get_true_w(len(v1))
    return np.sqrt(np.mean(np.square(v1 - v2)))


def find_best_alpha(lambda_):
    '''Find the best alpha value for a given lambda.'''
    training_sets = [[walk(5) for _ in xrange(10)] for _ in xrange(100)]
    alpha_range = np.arange(0.05, 0.55, 0.05)
    min_error = np.finfo('float').max
    best_alpha = alpha_range[0]
    for alpha in alpha_range:
        errors = []
        for training_set in training_sets:
            errors.append(rmse(batch_td(training_set, alpha=alpha, lambda_=lambda_)))
        error = np.mean(errors)
        if error < min_error:
            print alpha, error
            min_error = error
            best_alpha = alpha
    return best_alpha


def build_multi_plot(x, data, xlab, ylab):
    '''Build a plot with multiple line series.'''
    plt.ion()
    for label, y in data:
        plt.plot(x, y, marker='o', label=label)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.0,1.0))

    plt.show()


def build_plot(x, y, xlab, ylab):
    '''Build a plot with a single line series.'''
    plt.ion()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def fig3():
    '''Reproduce the results from Sutton's figure 3.'''
    true_w = get_true_w(5)
    training_sets = [[walk(5) for _ in xrange(10)] for _ in xrange(100)]
    lambda_range = np.arange(0.0, 1.1, 0.1)
    errors_by_lambda = []

    for lambda_ in lambda_range:
        errors = []
        for training_set in training_sets:
            w = batch_td(training_set, lambda_=lambda_, batch=True)
            errors.append(rmse(w))
        errors_by_lambda.append(np.mean(errors))
        print 'lambda:', lambda_, 'rmse:', errors_by_lambda[-1]

    build_plot(lambda_range, errors_by_lambda, 'lambda', 'RMSE')
    return lambda_range, errors_by_lambda


def fig4():
    '''Reproduce the results from Sutton's figure 4.'''
    true_w = get_true_w(5)
    training_sets = [[walk(5) for _ in xrange(10)] for _ in xrange(100)]
    lambda_range = (0.0, 0.3, 0.8, 1.0)
    alpha_range = np.arange(0.0, 0.7, 0.1)
    errors_by_lambda = []

    for lambda_ in lambda_range:
        errors_by_alpha = [] 
        for alpha in alpha_range:
            errors = []
            for training_set in training_sets:
                w = batch_td(training_set, lambda_=lambda_, alpha=alpha)
                errors.append(rmse(w))
            errors_by_alpha.append(np.mean(errors))
            print 'lambda:', lambda_, 'alpha:', alpha, 'rmse:', errors_by_alpha[-1]
        errors_by_lambda.append(('lambda={}'.format(lambda_), errors_by_alpha))

    build_multi_plot(alpha_range, errors_by_lambda, 'alpha', 'RMSE')
    return lambda_range, errors_by_lambda


def fig5():
    '''Reproduce the results from Sutton's figure 5.'''
    true_w = get_true_w(5)
    training_sets = [[walk(5) for _ in xrange(10)] for _ in xrange(100)]
    lambda_range = np.arange(0.0, 1.1, 0.1)
    # Best alphas taken from running the following
    # [find_best_alpha(l) for l in np.arange(0.0, 1.1, 0.1)]
    alpha_range = (0.25, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05)
    errors_by_lambda = []

    for lambda_, alpha in zip(lambda_range, alpha_range):
        errors = []
        for training_set in training_sets:
            w = batch_td(training_set, lambda_=lambda_, alpha=alpha)
            errors.append(rmse(w))
        errors_by_lambda.append(np.mean(errors))
        print 'lambda:', lambda_, 'rmse:', errors_by_lambda[-1]

    build_plot(lambda_range, errors_by_lambda, 'lambda', 'RMSE')
    return lambda_range, errors_by_lambda
