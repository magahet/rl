#!/usr/bin/env python


import numpy as np
from cvxopt import matrix, solvers

# Q = np.array([[0, -1, 1],
#               [1, 0, -1],
#               [-1, 1, 0]], dtype=np.double)

# Q = np.array([[-3, 2],
#               [4, -1]], dtype=np.double)

Q = np.array([
    [7., 3., -1.],
    [7., 3., 4.],
    [2., 2., 2.],
    [2., 2., 2.],
])


# Build constraints

G = np.vstack((
    np.concatenate((np.eye(Q.shape[0]), np.zeros((Q.shape[0], 1))), axis=1) * -1,
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

sol = solvers.lp(c, G, h, A, b)

print sol['x']
