#!/usr/bin/env python

import argparse
from soccer import Game
from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


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
        self.state_count = defaultdict(int)
        self.state_action_count = defaultdict(int)

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
        self.state_action_count[(state, action)] += 1
        last_q_value = self.get(state, action)
        V_next_state = np.max(self.get(next_state))
        q_value = (
            (1 - self.alpha) * last_q_value +
            self.alpha * ((1 - self.gamma) * reward +
                          self.gamma * V_next_state)
        )
        self.alpha = max(self.alpha * self.alpha_decay, self.min_alpha)
        if state == 'B21' and action == 1 and self.debug:
            print self.state_action_count.get(('B21', 1))
            # print alpha, reward, last_q_value, V_next_state, q_value
        self.set(state, action, q_value)

    def get_best_actions(self, state):
        self.state_count[state] += 1
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


class CEQAgent(object):
    def __init__(self, num_actions=5, num_players=2, debug=False):
        self.debug = debug
        self.Q = [{}, {}]
        self.gamma = 0.9
        self.actions = np.zeros((num_actions, num_actions))
        self.num_players = num_players
        self.state_count = defaultdict(int)

    def get(self, player, state, actions=None):
        if state in self.Q[player]:
            if actions is not None:
                Q = self.Q[player][state]
                return Q[actions]
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
        A1 = np.array(self.actions)
        A2 = np.array(self.actions)
        for i in np.eye(self.actions.shape[0], dtype=bool):
            A1[i] = np.sum(Q0[np.invert(i), :] - Q0[i, :], axis=0)
            A2[:, i] = np.vstack(np.sum(Q1[:, np.invert(i)] - Q1[:, i], axis=1))

        G = np.vstack(list(np.eye(self.actions.size) * -1) +  # each prob >= 0
                      [
                        A1.flatten(),  # rationality for P1
                        A2.flatten(),  # rationality for P2
                    ])

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
        # Maximize V
        # Constraints:
        #     Sum( probabilities action is chosen ) = 1
        #     For Each action J:
        #     Sum( Probability[i] * Q[ i, j ] ) - V >= 0
        pass

    def _friend(self, player, state):
        pass

    def update(self, state, actions, next_state, rewards):
        self.state_count[(state, actions)] += 1
        alpha = 1.0 / self.state_count[(state, actions)]

        for player in xrange(self.num_players):
            if self.policy == 'q':
                q_value = self.get(player, state, actions[player])
            else:
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
            if self.policy == 'q':
                self.set(player, state, actions[player], q_value)
            else:
                self.set(player, state, actions, q_value)

    def get_best_actions(self, state):
        return [np.random.randint(self.actions.shape[0]) for _ in xrange(2)]


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


def run_q(trials=10e5, dplot=False, debug=False):
    env = Game()
    agents = [QAgent(debug=debug) for _ in xrange(2)]
    x, y = [], []
    last = time.time()
    last_x = 0
    test_player = 0
    test_state = 'B21'
    test_action = 1
    done = True

    for episode in xrange(int(trials)):
        if done:
            state, rewards, done = env.reset()

        actions = [a.get_best_actions(state) for a in agents]
        next_state, rewards, done = env.step(actions)
        # print state, actions, next_state, rewards, done
        rewards = (rewards['A'], rewards['B'])

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
            print 100 * (episode / float(trials)), avg
            if dplot and len(x) > last_x:
                last_x == len(x)
                plot((x, y))

        state = next_state

    return agents, (x, y)


def run_ceq(trials=10e5, dplot=False, debug=False):
    env = Game()
    agent = QAgent(debug=debug)
    x, y = [], []
    last = time.time()
    last_x = 0
    test_player = 0
    test_state = 'B21'
    test_actions = (1, 4)

    state, rewards, done = env.reset()
    for episode in xrange(int(trials)):
        if done:
            print env.plot_grid()
            state, rewards, done = env.reset()

        actions = agent.get_best_actions(state)
        next_state, rewards, done = env.step(actions)
        rewards = (rewards['A'], rewards['B'])

        q = agent.get(test_player, state, actions)
        agent.update(state, actions, next_state, rewards)

        if state == test_state and actions in test_actions:
            delta = abs(q - agent.get(test_player,
                                      state, actions))
            if delta:
                x.append(episode)
                y.append(delta)

        if time.time() - last > 5:
            last = time.time()
            print 100 * (episode / float(trials)), len(y)
            if dplot and len(x) > last_x:
                last_x == len(x)
                plot((x, y))

        state = next_state

    return agent, (x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ceq.')
    parser.add_argument('policy')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-t', '--trials', type=int, default=10e5)
    args = parser.parse_args()
    func = {
        'q': run_q,
    }

    a, e = func[args.policy](args.trials, args.plot, args.debug)
    plot(e)
