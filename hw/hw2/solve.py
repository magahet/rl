#!/usr/bin/env python

import sympy as sp


def solve_poly(v1, v2, v3, v4, v5, v6, every=False):
    x = sp.symbols('x')
    e1, e2, e3, e4, e5, e6 = sp.symbols('e1:7')
    f = ((1-x)*(e1+e2*x+e3*x**2+e4*x**3+e5*x**4) +
         e6*(1-(1-x)*(1+x+x**2+x**3+x**4)) - e6)
    f2 = f.subs({e1: v1, e2: v2, e3: v3, e4: v4, e5: v5, e6: v6})
    roots = sp.solve(f2, x)
    solutions = [i for i in [sp.re(v) for v in roots] if i > 0 and i < 1]
    print solutions
    if not solutions:
        return None
    elif every:
        return solutions
    else:
        return solutions[0]


def solve(prob1, v_est, reward, every=False):
    prob2 = 1 - prob1

    v1 = (prob1 * (reward[0] + v_est[1]) +
          prob2 * (reward[1] + v_est[2]))

    v2 = (prob1 * (reward[0] + reward[2] + v_est[3]) +
          prob2 * (reward[1] + reward[3] + v_est[3]))

    v3 = (prob1 * (reward[0] + reward[2] + reward[4] + v_est[4]) +
          prob2 * (reward[1] + reward[3] + reward[4] + v_est[4]))

    v4 = (prob1 * (reward[0] + reward[2] + sum(reward[4:6]) + v_est[5]) +
          prob2 * (reward[1] + reward[3] + sum(reward[4:6]) + v_est[5]))

    v5 = (prob1 * (reward[0] + reward[2] + sum(reward[4:7]) + v_est[6]) +
          prob2 * (reward[1] + reward[3] + sum(reward[4:7]) + v_est[6]))

    v6 = (prob1 * (reward[0] + reward[2] + sum(reward[4:7])) +
          prob2 * (reward[1] + reward[3] + sum(reward[4:7])))

    return solve_poly(v1, v2, v3, v4, v5, v6, every=every)


probToState1 = 0.5
valueEstimates = (0, 3, 8, 2, 1, 2, 0)
rewards = (0, 0, 0, 4, 1, 1, 1)
