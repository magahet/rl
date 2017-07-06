import numpy as np
import itertools
import sys
import yaml


class Bar(object):
    def __init__(self, yaml_path):
        with open(yaml_path) as file_:
            conf = yaml.load(file_)
            self._attendance = np.array(conf.get('attendance'))
            self._fight = conf.get('fight')
            self.max_unknown = conf.get('max_unknown')
            self.num_patrons = self._attendance.shape[1]

    def nights(self):
        for group, fight in zip(self._attendance, self._fight):
            yield self._to_ints(group), fight

    @staticmethod
    def _to_ints(row):
        return set(np.where(row == True)[0])


class Agent(object):
    def __init__(self, bar):
        self.bar = bar
        self.H = set([c for c in itertools.permutations(range(self.bar.num_patrons), 2)])

    def run(self):
        output = []
        for group, fight in self.bar.nights():
            # print 'before: group={}, fight={}, H={}'.format(group, fight, self.H)
            L = self._get_L(group)
            if len(L) == 1:
                guess = L[0]
            else:
                guess = -1
                if fight:
                    self._filter_H(group)
            # print 'after: output={}, L={}, H={}'.format(guess, L, self.H)
            output.append(guess)
        return output

    def _is_fight(self, group, good=None, bad=None):
        return 1 if bad in group and good not in group else 0

    def _filter_H(self, group):
        '''Return subset of H with bad in group and good not in group.'''
        self.H = set([c for c in self.H if c[0] not in group and c[1] in group])

    def _get_L(self, group):
        return list(set([self._is_fight(group, h[0], h[1]) for h in self.H]))



if __name__ == '__main__':
    bar = Bar(sys.argv[1])
    agent = Agent(bar)
    print agent.run()
