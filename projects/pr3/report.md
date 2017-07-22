Project 3: RLDM - CS 7642
=====================================

GID: mmendiola3

Presentation: [https://youtu.be/](https://youtu.be/)

## Overview

This paper will review the reproduction of experiments done in "Correlated Q-Learning" [@Greenwald:2003:CL:3041838.3041869]. We will explore the agents designed to learn policies for the multi-agent soccer environment outlined in the original paper, and compare the result. Finally, we will discuss the challenges and assumptions made in order to reproduce the graphs in Figure 3.

## Soccer Environment
The soccer experiment rules were reproduced with a basic environment class built with an interface similar to those in OpenAI gym environments. It maintains state and models the actions and rewards outlined in the original paper. Experiments with various agents are able to make calls to env.step(actions) and receive back next state, rewards, and the done flag. Actions, state, and rewards includes data on both players in the soccer environment.

## Experiment
Using the soccer environment model, experiments were run with four learning agents (uCE-Q, Foe-Q, Friend-Q, and Q-learning). Each experiment was run over 10e5 steps (env steps, not episodes). The change in Q values for a specific player, state, and action(s) was recorded over these trials. This state is illustrated in Figure 4 of the original paper and the action(s) were: A=South, B=Stick. Points are plotted only on steps where this state, action(s) pair is observed, which is the first of many details left out of the original paper.

## Agents
Each learning agent used discount factor $\gamma$ of 0.9. They all used the same update procedure in Table 1 of [@Greenwald:2003:CL:3041838.3041869], including the Q value update formula. Each agent provided their own value function for $V_i(s')$.

### Q-Learner
The Q-learning experiment used two separate agents and Q tables for each player. The value update function was $V^*(s) = \underset{a \in A(s)}{\mathrm{max}} Q^*(s, a)$, which is simply the max Q value in a given state across all actions. In Q-learning, the agents only track Q values for their own actions.

Initial $\alpha$ was set to 0.3, and decayed exponentially with a decay factor of $10^{log(0.001)/5e6}$. The decay factor was inspired by [@littman01a] and both of these hyper-parameters required experimentation to find appropriate values. [@Greenwald:2003:CL:3041838.3041869] only specifies that $\alpha$ should decay to a minimum value of 0.001. The same decay schedule was used on $\epsilon$. The Q-learning agent was the only one run on-policy, with $\epsilon$-greedy actions. All other agents choose actions at random.

### Multi-Agent Learners
Foe-Q, Friend-Q, and uCE-Q all used an initial $\alpha$ of 1.0. Friend-Q used a decay schedule explained in [@greenwald2005correlated] as $\alpha = 1 / n(s,a)$. All other agents used the same exponential decay as Q-learning, with a decay rate of $10^{log(0.001)/4e6}$.

**Foe-Q** --
The value function for Foe-Q was taken from equation 5 in [@Greenwald:2003:CL:3041838.3041869]. Linear programming was used to find this value with probability constraints on the mixed strategy distribution (sum = 1, each $\geq$ 0) across the player's actions. Minimax constraints were modeled as $V - \sum_{a_1 \in A_1} \sigma_1(a_1)Q(s, a_1, a_2) \leq 0$ $\forall a_2 \in A_2(s)$, with the objective function to maximize V.

**Friend-Q** --
The value function for Friend-Q was simply the max Q value across all joint actions.

**uCE-Q** --
The correlated equilibrium was calculated similarly to Foe-Q, with the same probability constraints. Although CE strategy distributions are calculated across all joint actions. Rationality constraints were developed with the following code:

    for i in np.eye(num_actions, dtype=bool):
        A1[i] = np.sum(Q0[np.invert(i), :] - Q0[i, :], axis=0)
        A2[:, i] = np.vstack(np.sum(Q1[:, np.invert(i)] - Q1[:, i], axis=1))

This implements equation 2 in [@greenwald2005correlated], which states that each player's recommended action is at least as good as any other action given that the other player also follows the recommended strategy. This is expressed more concisely in the following constraints derived from [@Cunha13]:$\underset{\bar{s} \in S_{-p}}{\sum} (u^p_{j, \bar{s}} - u^p_{i, \bar{s}}) x_{i, \bar{s}} \leq 0$ $\forall p \text{ and } \forall i,j \in S_p$. Note that the inequality is reversed to act as an upper bound for the LP solver. The objective function is taken from equation 9 in [@Greenwald:2003:CL:3041838.3041869], which maximizes the sum of all players' rewards. The value function uses the optimal mixed strategy distribution from LP to return $\underset{\vec{a} \in A}{\sum} \sigma(\vec{a}) Q_i(s, \vec{a})$. The single greatest challenge in reproducing the original experiments was modeling the rationality constraints of CE for the LP solver.

## Results

![ce\label{ce}](fig/ce2.png){#id .class width=25%}
![foe\label{foe}](fig/foe2.png){#id .class width=25%}
![friend\label{friend}](fig/friend3.png){#id .class width=25%}
![q\label{q}](fig/q.png){#id .class width=25%}

The plots of Q value changes for the specifically chosen player, state, action(s) closely resemble those in the original experiment. They are not perfectly aligned, as the hyper-parameters and decay schedules were not fully disclosed and could not be perfectly reproduced. Experimentation with these parameters was required for each learner to reproduce similar rates of convergence as the original paper. However, the overall behavior of the various agents were replicated completely. Correlated-Q and Foe-Q had similar convergence rates. Although they did not produce identical graphs due to them running in at each step simply due to not statically seeding the stochastic environment. Friend-Q converged very quickly to a deterministic policy, and the Q-learner never converged as Q value deltas remained as high as the learning rate.


## References
