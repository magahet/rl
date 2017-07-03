import numpy as np
import matplotlib.pyplot as plt
import time


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
    def __init__(self, e=0.01, softmax=False, tau=0.5):
        self.e = e
        self.softmax = softmax
        self.tau = tau

    def choose(self, q):
        if self.softmax:
            blah = np.exp(q / self.tau)
            weights = blah / np.sum(blah)
            return np.random.choice(len(q), p=weights)
        elif np.random.uniform() < self.e:
            return np.argmax(q)
        else:
            return np.random.choice(len(q))


def run_trial(arms, explorer, threshold=0.0001, initial_q=0.0, max_steps=1000000):
    q = np.ones(len(arms)) * initial_q
    k = np.zeros(len(arms))
    max_q = []
    steps = 0

    while steps < max_steps:
        steps += 1
        index = explorer.choose(q)
        k[index] += 1
        q[index] += (arms[index].pull() - q[index]) / k[index]
        max_q.append(np.max(q))
        
    return steps, np.mean(max_q)


results = {}

arms = [Arm('normal', mean=np.random.normal(5, 1), var=1.0) for _ in xrange(100)]

results['e=0.01'] = run_trial(arms, Explorer(e=0.01))
results['e=0.1'] = run_trial(arms, Explorer(e=0.1))
results['softmax'] = run_trial(arms, Explorer(softmax=True, tau=5.0))
results['optimistic'] = run_trial(arms, Explorer(e=0.1), initial_q=5.0)

print 'k=10', results


results = {}

arms = []
for i in xrange(10):
    if i % 2 == 0:
        arms.append(Arm('uniform', low=0, high=10))
    else:
        arms.append(Arm('normal', mean=np.random.normal(5, 1), var=5.0))

results['e=0.01'] = run_trial(arms, Explorer(e=0.01))
results['e=0.1'] = run_trial(arms, Explorer(e=0.1))
results['softmax'] = run_trial(arms, Explorer(softmax=True, tau=5.0))
results['optimistic'] = run_trial(arms, Explorer(e=0.1), initial_q=5.0)

print 'k=100', results
