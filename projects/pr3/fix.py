#!/usr/bin/env python

from game import Soccer
from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import random

solvers.options['show_progress'] = False
solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
np.random.seed(42)
random.seed(42)


class MAgent(object):
    def __init__(self, policy='ce', num_actions=5, num_players=2,
                 alpha=1.0, debug=False):
        self.policy = policy
        self.debug = debug
        self.Q = [{}, {}]
        self.gamma = 0.9
        self.alpha = 1.0
        self.min_alpha = 0.001
        self.alpha_decay = np.power(10, np.log(0.001) / 4e6)
        self.actions = np.zeros((num_actions, num_actions))
        self.num_players = num_players
        self.state_count = defaultdict(int)
        self.policy_func = {
            'ce': self._ce,
            'foe': self._foe,
            'friend': self._friend,
        }.get(policy)
        self.count = defaultdict(int)

    def get(self, player, state, actions=None):
        if state in self.Q[player]:
            if actions is not None:
                return self.Q[player][state][actions[0]][actions[1]]
            else:
                return self.Q[player][state]
        else:
            if actions is None:
                return np.array(self.actions)
            else:
                return 0.0

    def set(self, player, state, actions, value):
        if state not in self.Q[player]:
            self.Q[player][state] = np.array(self.actions)
        self.Q[player][state][actions[0]][actions[1]] = value

    def _ce(self, player, state):
        Q0 = self.get(0, state)
        Q1 = self.get(1, state)

        # Rationality constraints
        G_rationality = []
        for i in np.eye(self.actions.shape[0], dtype=bool):
            for j in np.eye(self.actions.shape[0], dtype=bool):
                if np.array_equal(i, j):
                    continue
                G0 = np.array(self.actions)
                G0[i] = Q0[j, :] - Q0[i, :]
                G1 = np.array(self.actions)
                G1[:, i] = np.vstack(Q1[:, j] - Q1[:, i])
                G_rationality.append(G0.flatten())
                G_rationality.append(G1.flatten())

        # each prob >= 0
        G_prob = list(np.eye(self.actions.size) * -1)
        G = np.vstack(G_prob + G_rationality)

        h = np.zeros(len(G))
        A = np.vstack([
            np.ones(self.actions.size),  # sum of probs = 1
        ])
        b = np.ones(1)

        print len(G) + len(A)

        c = matrix((Q0 + Q1).flatten() * -1)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        sol = solvers.lp(c, G, h, A, b, solver='cvxopt_glpk')
        x = sol['x']
        print 'Q0'
        print Q0
        print 'Q1'
        print Q1
        print 'G'
        print G
        print 'h'
        print h
        print 'A'
        print A
        print 'b'
        print b
        print 'c'
        print c
        print 'x'
        print x

        p = np.array(x).reshape(self.actions.shape)
        v = np.sum(self.get(player, state) * p)
        print 'ce value', v
        return v

    def _foe(self, player, state):
        Q = self.get(player, state)

        G = np.vstack((
            np.concatenate((np.eye(Q.shape[0]),
                            np.zeros((Q.shape[0], 1))), axis=1) * -1,
            np.concatenate((Q.T * -1, np.ones((Q.shape[1], 1))), axis=1)
        ))
        h = np.zeros(len(G))
        A = np.ones((1, len(Q) + 1))
        A[0][-1] = 0
        b = np.ones(1)

        c = np.zeros(len(Q) + 1)
        c[-1] = -1

        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        c = matrix(c)

        sol = solvers.lp(c, G, h, A, b, solver='glpk')
        x = sol.get('x')

        v = x[-1]
        if player == 0:
            print 'foe value', v
            return v
        else:
            print 'foe value', -v
            return -v

    def _friend(self, player, state):
        return np.max(self.get(player, state))

    def update(self, state, actions, next_state, rewards):
        self.count[(state, actions)] += 1
        if self.policy == 'friend':
            self.alpha = 1.0 / self.count[(state, actions)]
        for player in xrange(self.num_players):
            q_value = self.get(player, state, actions)

            if np.sum(self.get(player, next_state)) == 0:
                V_next_state = 0.0
            else:
                V_next_state = self.policy_func(player, next_state)

            q_value = (
                (1 - self.alpha) * q_value +
                self.alpha * ((1 - self.gamma) * rewards[player] +
                              self.gamma * V_next_state)
            )
            self.set(player, state, actions, q_value)

        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)

    def get_best_actions(self, state):
        return tuple([
            np.random.randint(self.actions.shape[0]) for _ in xrange(2)
        ])


def plot(data):
    x, y = data
    plt.ion()
    plt.ylim((0, 0.5))
    plt.ylabel('Q-value Difference')
    plt.xlabel('Simulation Iteration')
    plt.plot(x, y, linewidth=0.2, color='black')
    plt.pause(0.05)


def save(data, path):
    with open(path, 'wb') as file_:
        pickle.dump(data, file_)


def load(path):
    with open(path, 'rb') as file_:
        return pickle.load(file_)


env = Soccer()
agent1 = MAgent(policy='ce')
agent2 = MAgent(policy='foe')
done = True

while True:
    if done:
        state, rewards, done = env.reset()

    actions = tuple(random.sample(range(5), 2))

    next_state, rewards, done = env.step(actions)

    agent1.update(state, actions, next_state, rewards)
    agent2.update(state, actions, next_state, rewards)

    state = next_state

    q1 = np.sum(agent1.Q[0].values())
    q2 = np.sum(agent2.Q[0].values())

    print q1, q2

    if (q1 - q2) > 1:
        break
