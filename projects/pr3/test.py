#!/usr/bin/env python


import sys
import numpy as np
from cvxopt import matrix, solvers

Q1 = np.array([[6, 2], [7, 0]], dtype=np.double)
Q2 = np.array([[6, 7], [2, 0]], dtype=np.double)


Q1 = np.array([[6, 2, 0], [7, 0, 0], [0, 0, 0]], dtype=np.double)
Q2 = np.array([[6, 7, 0], [2, 0, 0], [0, 0, 0]], dtype=np.double)

Q1 = np.zeros((3, 3))
Q2 = np.zeros((3, 3))

# Build constraints

# Rationality constraints
A1 = np.zeros(Q1.shape)
A2 = np.zeros(Q1.shape)
for i in np.eye(Q1.shape[0], dtype=bool):
    A1[i] = np.sum(Q1[np.invert(i), :] - Q1[i, :], axis=0)
    A2[:, i] = np.vstack(np.sum(Q2[:, np.invert(i)] - Q2[:, i], axis=1))

# Probs sum to one
A = np.vstack(list(np.eye(Q1.size) * -1) +  # each prob >= 0
              [
                A1.flatten(),  # rationality for P1
                A2.flatten(),  # rationality for P2
                np.ones(Q1.size),  # sum of probs <= 1
                np.ones(Q1.size) * -1,  # sum of probs >= 1
             ])
b = np.zeros(len(A))
b[-2:] = 1
A = matrix(A)
b = matrix(b)
c = matrix((Q1 + Q2).flatten() * -1)

print A
print b
print c

sol = solvers.lp(c, A, b)

print sol['x']
sys.exit()
