#!/usr/bin/env python


import numpy as np
from cvxopt import matrix, solvers


Q1 = np.array([[6, 2, 0], [7, 0, 0], [0, 0, 0]], dtype=np.double)
Q2 = np.array([[6, 7, 0], [2, 0, 0], [0, 0, 0]], dtype=np.double)

Q1 = np.array([[6, 2], [7, 0]], dtype=np.double)
Q2 = np.array([[6, 7], [2, 0]], dtype=np.double)

# Build constraints

# Rationality constraints
A1 = np.zeros((2, 2))
A2 = np.zeros((2, 2))
for i in np.eye(2, dtype=bool):
    A1[i] = Q1[np.invert(i), :] - Q1[i, :]
    A2[:, i] = Q2[:, np.invert(i)] - Q2[:, i]

# Probs sum to one
A = np.vstack(list(np.eye(4) * -1) +  # each prob >= 0
              [
                A1.flatten(),  # rationality for P1
                A2.flatten(),  # rationality for P2
                np.ones((1, 4)),  # sum of probs <= 1
                np.ones((1, 4)) * -1,  # sum of probs >= 1
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
