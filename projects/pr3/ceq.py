#!/usr/bin/env python

import argparse
from game import Soccer
from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
np.random.seed(42)


class QAgent(object):
    def __init__(self, num_actions=5, debug=False):
        self.debug = debug
        self.Q = {}
        self.gamma = 0.9
        self.alpha = 0.3
        self.epsilon = 0.2
        self.min_alpha = 0.001
        self.min_epsilon = 0.001
        self.alpha_decay = np.power(10, np.log(0.001) / 5e6)
        self.epsilon_decay = np.power(10, np.log(0.001) / 5e6)
        self.actions = np.zeros(num_actions)

    def get(self, state, action=None):
        if state in self.Q:
            if action is not None:
                return self.Q[state][action]
            else:
                return self.Q[state]
        else:
            if action is None:
                return np.array(self.actions)
            else:
                return 0.0

    def set(self, state, action, value):
        if state not in self.Q:
            self.Q[state] = np.array(self.actions)
        self.Q[state][action] = value

    def update(self, state, action, next_state, reward):
        last_q_value = self.get(state, action)
        V_next_state = np.max(self.get(next_state))
        q_value = (
            (1 - self.alpha) * last_q_value +
            self.alpha * ((1 - self.gamma) * reward +
                          self.gamma * V_next_state)
        )
        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)
        self.set(state, action, q_value)

    def get_best_actions(self, state):
        greedy = (
            state in self.Q and
            np.random.uniform() > self.epsilon and
            np.sum(self.Q[state]) > 0
        )
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        if greedy:
            return np.argmax(self.get(state))
        else:
            return np.random.randint(self.actions.shape[0])


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

        c = matrix((Q0 + Q1).flatten() * -1)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        sol = solvers.lp(c, G, h, A, b, solver='cvxopt_glpk')

        p = np.array(sol['x']).reshape(self.actions.shape)
        return np.sum(self.get(player, state) * p)

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

        sol = solvers.lp(c, G, h, A, b, solver='cvxopt_glpk')
        x = sol.get('x')
        v = x[-1] if x is not None else 0.0
        if player == 0:
            return v
        else:
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


def plot(data, title):
    x, y = data
    plt.ion()
    plt.ylim((0, 0.5))
    plt.title(title)
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


def run_q(args, title):
    env = Soccer()
    agents = [QAgent(debug=args.debug) for _ in xrange(2)]
    x, y = [], []
    last = time.time()
    last_x = 0
    test_player = 0
    test_state = '[(2, 1), (1, 1)],1'
    test_action = 2
    done = True

    for episode in xrange(int(args.trials)):
        if done:
            state, rewards, done = env.reset()

        actions = [a.get_best_actions(state) for a in agents]
        next_state, rewards, done = env.step(actions)

        q1 = agents[test_player].get(state, actions[test_player])
        for num, agent in enumerate(agents):
            agent.update(state, actions[num], next_state, rewards[num])

        if state == test_state and actions[test_player] == test_action:
            q2 = agents[test_player].get(state, actions[test_player])
            delta = abs(q2 - q1)

            if delta:
                x.append(episode)
                y.append(delta)

        if time.time() - last > 10:
            last = time.time()
            avg = np.mean(y[-min(len(y), 100)]) if y else 0.0
            print 100 * (episode / float(args.trials)), avg
            if args.plot and len(x) > last_x:
                last_x == len(x)
                plot((x, y), title)

        state = next_state

    return agents, (x, y)


def run_m(args, title):
    env = Soccer()
    agent = MAgent(policy=args.policy, alpha=args.alpha, debug=args.debug)
    x, y = [], []
    last = time.time()
    last_x = 0
    test_player = 0
    test_state = '[(2, 1), (1, 1)],1'
    test_actions = (2, 4)
    done = True

    for episode in xrange(int(args.trials)):
        if done:
            state, rewards, done = env.reset()

        actions = agent.get_best_actions(state)
        next_state, rewards, done = env.step(actions)

        q1 = agent.get(test_player, state, actions)
        agent.update(state, actions, next_state, rewards)

        if state == test_state and actions == test_actions:
            q2 = agent.get(test_player, state, actions)
            delta = abs(q2 - q1)

            if delta:
                x.append(episode)
                y.append(delta)

        if time.time() - last > 10:
            last = time.time()
            avg = np.mean(y[-min(len(y), 100)]) if y else 0.0
            print 100 * (episode / float(args.trials)), avg
            if args.plot and len(x) > last_x:
                last_x == len(x)
                plot((x, y), title)

        state = next_state

    return agent, (x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ceq.')
    parser.add_argument('policy')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-t', '--trials', type=int, default=10e5)
    parser.add_argument('-a', '--alpha', type=float, default=1.0)
    args = parser.parse_args()
    func = {
        'q': run_q,
        'ce': run_m,
        'foe': run_m,
        'friend': run_m,
    }

    title = {
        'ceq': 'Correlated-Q',
        'foe': 'Foe-Q',
        'friend': 'Friend-Q',
    }.get(args.policy)
    a, e = func[args.policy](args, title)
    plot(e, title)
