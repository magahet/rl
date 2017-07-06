import numpy as np
import matplotlib.pyplot as plt
import sys


class Arm(object):
    def __init__(self, type_, low=None, high=None, mean=None, var=None):
        self.type = type_
        self.low = low
        self.high = high
        self.mean = mean
        if var:
            self.std = np.sqrt(var)

    def pull(self):
        if self.type == 'uniform':
            return np.random.uniform(self.low, self.high)
        elif self.type == 'normal':
            return np.random.normal(self.mean, self.std)


class Explorer(object):
    def __init__(self, e=0.01, softmax=False, tau=0.5, initial_q=0.0):
        self.e = e
        self.softmax = softmax
        self.tau = tau
        self.initial_q = initial_q

    def __str__(self):
        if self.softmax:
            return 'softmax--tau={}'.format(self.tau)
        elif self.initial_q != 0.0:
            return 'optimistic--e={}'.format(self.e)
        else:
            return 'epsilon-greedy--e={}'.format(self.e)

    def choose(self, q):
        if self.softmax:
            blah = np.exp(q / self.tau)
            weights = blah / np.sum(blah)
            return np.random.choice(len(q), p=weights)
        elif np.random.uniform() < self.e:
            return np.argmax(q)
        else:
            return np.random.choice(len(q))


def run_trial(arms, explorer, pulls, bandits):
    avg_reward = np.zeros(pulls)

    q = np.ones((bandits, len(arms))) * explorer.initial_q
    k = np.zeros((bandits, len(arms)))

    for pull_index in xrange(pulls):
        if pull_index % (pulls / 100) == 0:
            print pull_index
        rewards = np.zeros(bandits)
        for bandit_index in xrange(bandits):
            arm_index = explorer.choose(q[bandit_index])
            reward = arms[arm_index].pull()
            rewards[bandit_index] = reward
            k[bandit_index][arm_index] += 1
            q[bandit_index][arm_index] += (reward - q[bandit_index][arm_index]) / k[bandit_index][arm_index]
        avg_reward[pull_index] = np.mean(rewards)
    return avg_reward


def test_explorer(arms, explorer, pulls, bandits):
    title = 'k={}, {}'.format(len(arms), explorer)
    avg_reward = run_trial(arms, explorer, pulls, bandits)
    plt.figure(title)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Pulls')
    plt.ylabel('Avg. Max Q')
    plt.plot(avg_reward)
    plt.savefig('k{}-{}.png'.format(len(arms), explorer))


arms_100 = [Arm('normal', mean=np.random.normal(5, 1), var=1.0) for _ in xrange(100)]

arms_10 = []
for i in xrange(10):
    if i % 2 == 0:
        arms_10.append(Arm('uniform', low=0, high=10))
    else:
        arms_10.append(Arm('normal', mean=np.random.normal(5, 1), var=5.0))

explorers = {
    'e001': Explorer(e=0.01),
    'e01': Explorer(e=0.1),
    'soft03': Explorer(softmax=True, tau=0.3),
    'soft05': Explorer(softmax=True, tau=0.5),
    'opt': Explorer(e=0.1, initial_q=15.0),
}

pulls = {
    100: 50000,
    10: 20000,
}

arms = {
    100: arms_100,
    10: arms_10
}

arm_key = int(sys.argv[1])
explorer = explorers.get(sys.argv[2])
#bandits = 20
bandits = 2000

test_explorer(arms.get(arm_key), explorer, pulls.get(arm_key), bandits)
