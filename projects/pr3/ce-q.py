#!/usr/bin/env python

from soccer import Game
from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time


class QAgent(object):
    def __init__(self, num_actions=5):
        self.Q = {}
        self.state_count = defaultdict(int)
        self.gamma = 0.99
        self.epsilon = 0.01
        self.num_actions = num_actions

    def update(self, state, action, next_state, reward):
        self.state_count[(state, action)] += 1

        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(self.num_actions)

        old_Q = self.Q[state][action]

        alpha = 1.0 / self.state_count[(state, action)]
        self.Q[state][action] = (
            (1 - alpha) * self.Q[state][action] +
            alpha * (reward + self.gamma * np.max(self.Q[next_state]))
        )
        return abs(self.Q[state][action] - old_Q)

    def get_best_action(self, state):
        greedy = (
            state in self.Q and
            np.random.uniform() > self.epsilon and
            np.sum(self.Q[state]) > 0
        )
        if greedy:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(self.num_actions)


class CEQAgent(object):
    def __init__(self, num_actions=5, num_players=2, policy='u'):
        self.Q = [{}, {}]
        self.alpha = 0.001
        self.gamma = 0.9
        self.epsilon = 0.001
        self.num_actions = num_actions
        self.num_players = num_players
        self.policy_func = {
            'u': self._utilitarian,
            'e': self._egalitarian,
            'r': self._republican,
            'l': self._libertarian,
        }.get(policy)
        self.action_permutations = [
            i for i in itertools.permutations(range(num_actions), num_players)
        ]

    def _utilitarian(self, Q):
        '''Maximize sum of all rewards.'''
        return np.concatenate(Q).flatten() * -1

    def _egalitarian(self, state):
        pass

    def _republican(self, state):
        pass

    def _libertarian(self, state):
        pass

    def ce_value(self, player, state):
        Q = [self.Q[i][state] for i in self.num_players]
        # Get objective function based on policy
        c = self.policy_func(Q)

        # Make the sum of probabilities == 1
        A_eq = np.ones((1, len(self.action_permutations)))
        b_eq = np.array([1.0])

        # Build upper bound constraints
        A_ub = []

        # Make each probability >= 0
        A_ub.extend(list(np.eye(self.num_actions) * -1))

        # Build rationality constraints




    def update(self, state, actions, next_state, rewards):
        a1, a2 = actions
        for player in xrange(self.num_players):
            if state not in self.Q[player]:
                self.Q[player][state] = np.zeros((self.num_actions,
                                                  self.num_actions))
            if next_state not in self.Q:
                self.Q[player][next_state] = np.zeros((self.num_actions,
                                                       self.num_actions))
        old_Q = self.Q[0][state][a1][a2]

        for player in xrange(self.num_players):
            V = self.ce_value(player, next_state)

            self.Q[player][state][a1][a2] += (
                (1 - self.alpha) * self.Q[player][state][a1][a2] +
                self.alpha * (1 - self.gamma) * rewards[player] +
                self.gamma * V
            )

        return abs(self.Q[0][state][a1][a2] - old_Q)

    def get_best_actions(self, state):
        actions = []
        for player in xrange(self.num_players):
            greedy = (
                state in self.Q[player] and
                np.random.uniform() > self.epsilon and
                np.sum(self.Q[player][state]) > 0
            )
            if greedy:
                actions.append(
                    np.argmax(self.Q[player][state]) % self.num_actions)
            else:
                actions.append(np.random.randint(self.num_actions))
        return tuple(actions)


def plot(data):
    plt.plot(data, linewidth=1, color='black')
    plt.pause(0.05)


def save(data, path):
    with open(path, 'wb') as file_:
        pickle.dump(data, file_)


def load(path):
    with open(path, 'rb') as file_:
        return pickle.load(file_)


def run_q():
    trials = 10e5
    env = Game()
    Qa = QAgent()
    Qb = QAgent()
    error_by_trial = []
    last = time.time()

    for episode in xrange(int(trials)):
        max_error = 0.0
        state, rewards, done = env.reset()

        while not done:
            action_a = Qa.get_best_action(state)
            action_b = Qb.get_best_action(state)
            next_state, rewards, done = env.step((action_a, action_b))
            error = Qa.update(state, action_a, next_state, rewards.get('A'))
            max_error = error if error > max_error else max_error
            Qb.update(state, action_b, next_state, rewards.get('B'))
            state = next_state
            # env.plot_grid()

        if time.time() - last > 5:
            last = time.time()
            print 100 * (episode / trials), max_error
            # plot(error_by_trial)

        error_by_trial.append(max_error)

    return error_by_trial


def run_ceq():
    trials = 10e5
    env = Game()
    Q = CEQAgent()
    error_by_trial = []
    last = time.time()

    for episode in xrange(int(trials)):
        error = 0.0
        state, rewards, done = env.reset()

        while not done:
            actions = Q.get_best_actions(state)
            next_state, rewards, done = env.step(actions)
            rewards = (rewards['A'], rewards['B'])
            error += Q.update(state, actions, next_state, rewards)
            state = next_state
            # env.plot_grid()

        if time.time() - last > 5:
            last = time.time()
            print 100 * (episode / trials), error
            # plot(error_by_trial)

        error_by_trial.append(error)

    return error_by_trial

    # print "actions: [N: 0, S: 1, E: 2, W: 3, Stay: 4] \n"


if __name__ == '__main__':
    plt.ion()
    # error = run_ceq()
    error = run_q()
