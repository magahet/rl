#!/usr/bin/env python

from soccer import Game
from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers


class Agent(object):
    def __init__(self, num_actions=5, num_players=2, policy='q', debug=False):
        self.debug = debug
        self.Q = [{}, {}]
        self.alpha = 0.001
        self.gamma = 0.9
        self.epsilon = 0.001
        self.actions = np.zeros((num_actions, num_actions))
        self.num_players = num_players
        self.state_count = defaultdict(int)
        self.policy = policy
        self.policy_func = {
            'ce-q': self._ce,
            'foe-q': self._foe,
            'friend-q': self._friend,
            'q': self._q,
        }.get(policy)

    def get(self, player, state, actions=None):
        if state in self.Q[player]:
            if actions is not None:
                return self.Q[player][state][actions[0]][actions[1]]
            else:
                return self.Q[player][state]
        else:
            return 0.0

    def set(self, player, state, actions, value):
        if state not in self.Q[player]:
            self.Q[player][state] = np.array(self.actions)
        self.Q[player][state][actions[0]][actions[1]] = value

    def _q(self, player, state):
        return np.max(self.get(player, state))

    def _ce(self, player, state):
        Q0 = self.get(0, state)
        Q1 = self.get(1, state)
        # Rationality constraints
        A1 = np.array(self.actions)
        A2 = np.array(self.actions)
        for i in np.eye(self.actions.shape[0], dtype=bool):
            A1[i] = Q0[np.invert(i), :] - Q0[i, :]
            A2[:, i] = Q1[:, np.invert(i)] - Q1[:, i]

        # Probs sum to one
        A = np.vstack(list(np.eye(self.actions.size) * -1) +  # each prob >= 0
                      [
                        A1.flatten(),  # rationality for P1
                        A2.flatten(),  # rationality for P2
                        np.ones((1, self.actions.size)),  # sum of probs = 1
                        np.ones((1, self.actions.size)) * -1,
                    ])
        b = np.zeros(len(A))
        b[-2:] = 1
        A = matrix(A)
        b = matrix(b)
        c = matrix((Q0 + Q1).flatten() * -1)

        sol = solvers.lp(c, A, b)

        p = np.array(sol['x']).reshape(self.actions.shape)
        return np.sum(self.Q[player] * p)

    def _foe(self, player, state):
        pass

    def _friend(self, player, state):
        pass

    def update(self, state, actions, next_state, rewards):
        self.state_count[(state, actions)] += 1
        alpha = 1.0 / self.state_count[(state, actions)]

        for player in xrange(self.num_players):
            q_value = self.get(player, state, actions)
            V_next_state = self.policy_func(player, next_state)
            q_value = (
                (1 - alpha) * q_value +
                alpha * ((1 - self.gamma) * rewards[player] +
                         self.gamma * V_next_state)
            )
            if self.debug:
                print q_value, alpha, self.gamma, rewards[player], V_next_state
                print 'q', q_value, 'V', V_next_state, rewards
            self.set(player, state, actions, q_value)

    def get_best_actions(self, state):
        actions = []
        for player in xrange(self.num_players):
            greedy = (
                # TODO should non q agents have greedy actions?
                self.policy == 'q' and
                state in self.Q[player] and
                np.random.uniform() > self.epsilon and
                np.sum(self.Q[player][state]) > 0
            )
            if greedy:
                index = np.argmax(self.Q[player][state])
                index = np.unravel_index(index, self.actions.shape)
                actions.append(index[player])
            else:
                actions.append(np.random.randint(self.actions.shape[0]))
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


def run_trial(policy, trials=10e5):
    env = Game()
    agent = Agent(debug=False)
    x, y = [], []
    last = time.time()
    test_player = 0
    test_state = 'B21'
    test_actions = (1, 4)
    done = True

    for episode in xrange(int(trials)):
        if done:
            state, rewards, done = env.reset()
        actions = agent.get_best_actions(state)
        next_state, rewards, done = env.step(actions)
        rewards = (rewards['A'], rewards['B'])

        q = agent.get(test_player, test_state, test_actions)
        agent.update(state, actions, next_state, rewards)

        if state == test_state and actions == test_actions:
            delta = abs(q - agent.get(test_player, test_state, test_actions))
            print delta
            x.append(episode)
            y.append(delta)

        state = next_state
        # print actions
        # env.plot_grid()

    if time.time() - last > 5:
        last = time.time()
        print 100 * (episode / float(trials))
        # plot(error_by_trial)

    # print agent.Q[0].keys()

    return agent, (x, y)

    # print "actions: [N: 0, S: 1, E: 2, W: 3, Stay: 4] \n"


if __name__ == '__main__':
    plt.ion()
    # error = run_trial('q')
