#!/usr/bin/env python

from soccer import Game
import numpy as np
import matplotlib as plt


def create_state_comb(p_a_states, p_b_states):
    """ Creates a dictionary that represents the state space possible
    combinations.

    Args:
        p_a_states (list): List with the numerical state labels for player A
        p_b_states (list): List with the numerical state labels for player B

    Returns:
        dict: Dictionary with the state space representation. Each element is
        labeled using the format [XYZ] where:
                - X: shows who has the ball, either A or B.
                - Y: state where player A is.
                - Z: state where player B is.

            The key values hold a numeric value using the counter id_q.

    """

    states = {}
    ball_pos = ['A', 'B']
    id_q = 0

    for b in ball_pos:

        for p_a in p_a_states:

            for p_b in p_b_states:

                if p_a != p_b:
                    states[b + str(p_a) + str(p_b)] = id_q
                    id_q += 1

    return states


def print_status(goal, new_state, rewards, total_states):
    print ""
    print "Players state label: {}".format(new_state)
    print "Players state in the numerical table: {}".format(
        total_states[new_state])
    print "Rewards for each player after move: {}".format(rewards)
    print "Goal status: {}".format(goal)
    print "-" * 20 + "\n"


class QAgent(object):
    def __init__(self, num_actions=5):
        self.Q = {}
        self.alpha = 0.001
        self.gamma = 0.99
        self.epsilon = 0.01
        self.num_actions = num_actions

    def update(self, state, action, next_state, reward):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(self.num_actions)

        old_Q = self.Q[state][action]

        self.Q[state][action] += (
            self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) -
                          self.Q[state][action])
        )
        return abs(self.Q[state][action] - old_Q)

    def get_best_action(self, state):
        greedy = (
            state in self.Q and
            np.random.uniform > self.epsilon and
            np.sum(self.Q[state]) > 0
        )
        if greedy:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(self.num_actions)


def plot(data):
    plt.plot(data)


def main():
    trials = 10e5
    env = Game()
    Qa = QAgent()
    Qb = QAgent()
    error_by_trial = []

    for episode in xrange(int(trials)):
        error = 0.0
        state, rewards, done = env.reset()

        if episode % (trials / 100) == 0:
            print 100 * (episode / trials)

        while not done:
            action_a = Qa.get_best_action(state)
            action_b = Qb.get_best_action(state)
            next_state, rewards, done = env.step((action_a, action_b))
            error += Qa.update(state, action_a, next_state, rewards.get('A'))
            Qb.update(state, action_b, next_state, rewards.get('B'))
            state = next_state
            # env.plot_grid()

        error_by_trial.append(error)

    return error_by_trial

    # print "actions: [N: 0, S: 1, E: 2, W: 3, Stay: 4] \n"


if __name__ == '__main__':
    error = main()
