import numpy as np
import itertools


numPatrons = 5
good, bad = np.random.choice(5, size=2, replace=False)


def next_():
    group = np.zeros(numPatrons)
    while np.sum(group) < 2:
        group = np.random.choice([0, 1], size=numPatrons)
    return group


def to_ints(row):
    return set(np.where(row > 0)[0])


def get_L(group):
    return set([h for h in H if set(h).issubset(group)])


def is_fight(group, good, bad):
    return bad in group and good not in group


def filter_H(H, group):
    '''Return subset of H with bad in group and good not in group.'''
    return set([c for c in H if c[0] not in group and c[1] in group])


print good, bad

H = set([c for c in itertools.permutations(range(numPatrons), 2)])

while len(H) > 1:
    group = to_ints(next_())
    L = get_L(group)
    if len(L) == 1:
        h_good, h_bad = L[0]
        guess = is_fight(group, h_good, h_bad)
    else:
        fight = is_fight(group, good, bad)
        guess = None
        if fight:
            H = filter_H(H, group)
        
    print guess, len(H)
