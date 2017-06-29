Project 2: RLDM - CS 7642
=====================================

The reward for each training episode while training your agent 
The reward per trial for 100 trials using your trained agent 
The effect of hyper-parameters (alpha, lambda, epsilon) on your agent 
    You pick the ranges 
    Be prepared to explain why you chose them 
Discuss your results 
Explain the algorithms you selected and why you did 
Describe any problems/pitfalls you encountered 
Explain your experiments 

GID: mmendiola3

Presentation: [https://youtu.be/](https://youtu.be/)

## Overview

This paper will review an RL agent designed to solve the LunarLander-v2 problem. We will review the algorithms and hyper-parameters chosen and present how these decisions affected performance of the agent. We will also address the many pitfalls encountered while constructing the RL agent.


## Agent Design

The base algorithm used by the agent is Q-learning, with a neural network for function approximation. My initial thought was to descretize the state space to allow for a simple Q table implementation. This was quickly abandoned in favor of the more generalized, and adaptable, function approximation.

DQN (Deep Q Network) refers to a Q-learning agent with a neural network for learning and predicting Q values at given (state, action) pairs. The 'Deep" part refers to a network with many hidden layers, as was used in the original Deep Mind research. The agent used to solve LunarLander uses a single hidden dense layer with 400 nodes. This was enough to approximate Q values for this environment without requiring longer training times on a more complex network.

I tested many combinations of layers and nodes to find the simplest network capable of approximating Q values with enough granularity to guide the lander towards the goal. I tested single hidden layers with nodes between 16 and 1024 and two hidden layer networks with the second layer either equal or half that of the fist layer. Each of these experiments resulted in worse or similar initial performance or increased training time.

Thekkk

### Error Measure
Each experiment measured the accuracy of the algorithm to predict the probabilities of right-side termination at each of the five states. The RMS error was calculated as $sqrt(mean(square(predicted\_weights - actual\_weights)))$, with actual weights of (1/6, 2/6, 3/6, 4/6, 5/6). Error was averaged over repeated experiments on 100 state sequences.

### Weight Updates
Both experiments used Sutton's $\Delta w_t$ update equation from (4), $\Delta w_t = \alpha (P_{t+1} - P_t) \sum_{k=1}^t \lambda^{t-k} \nabla_w P_k$. The code uses an eligibility vector to keep track of the exponential weighting with recency. This method of incremental computation is shown in Sutton's paper as $e_{t+1} = \nabla_w P_{t+1} + \lambda e_t$. In code, this is accomplished with $e = e + x_{t-1}$ before the weight update, and $e = \lambda e$ after. Each state's prediction value is calculated as $P_t = w^T x_t$, except for the terminal state, which is given the scalar value 0 or 1. The resulting $\Delta w_t$ update equation in code is $\Delta w_t = \alpha (P_{t+1} - P_t) e$.

## Batch TD($\lambda$)


### Repeated Presentation
For the first experiment, each $\Delta w_t$ was accumulated into a single running $\Delta w$ vector. At the end of all episodes, $\Delta_w$ was added to the weight vector, then reset for the next round of training. These training sets were presented repeatedly until the change in w reached a given threshold. Sutton's paper did not provide an explicit stopping criteria; For this experiment, training ended when $|\Delta w|_2$ fell below 1e-7. By that point, changes in error from actual weights are negligable. 

### Best $\alpha$
Sutton mentions using the best $\alpha$ to show error accross various lambda values in his Figure 3. This seems to be contridicted in a later statement, "For small $\alpha$, the weight vector always converged in this way, and always to the same final value." This implies that as long as $\alpha$ is small enough, the resulting error for a given lambda will be the same. Through experimentation, I discovered this to be accurate. While larger $\alpha$ reduced time to convergence, too large of an $\alpha$ caused the weight vector to diverge. Figure \ref{fig3} was constructed by running the average of 100 sequences of 10 on the batch TD algorithm for lambda values from 0 to 1, using an $\alpha$ of 0.005.

### Results
![fig3\label{fig3}](fig/fig3.png){#id .class width=60%}

Interestingly, Figure \ref{fig3} resembles a combination of Sutton's original Figure 3 and the one in his Erratum. The results here are monotonically increasing (like Erratum), but resemble the scale from the original figure. I am not able to attribute the scale difference between Figure \ref{fig3} and the Erratum figure to any particular parameter. Especially given Sutton's assertion that the weight vector would always converge to the same final value.

## Online TD($\lambda$)

![fig4\label{fig4}](fig/fig4.png){#id .class width=50%}
![fig5\label{fig5}](fig/fig5.png){#id .class width=50%}

### Sequential Weight Updates
The second experiment does a single passthrough of the 10 sequences, and updates the weight vector after each state sequence. This removes the need to monitor the process of convergence, and greatly reduces the number of weight update calculations.

### Results
The left-hand figure reproduces the result from Sutton's Figure 4. In this plot, we measure the error given various $\lambda$ and $\alpha$ values. The general scale, and relationship between $\lambda$ values, are similar to Sutton's result. A few exceptions include $\lambda = 0$, where error after $\alpha = 0.4$ increase at a much higher rate. Also, $\lambda = 0.3$ and $\lambda = 0.8$ crossover after $\alpha = 0.5$. These results still support Sutton's assertion that testing error (not training error) is highest with $\lambda = 1$ (Widrow-Hoff) for various values of $\alpha$.

The right-hand plot shows that, given the optimal $\alpha$ for each value of $\lambda$, the optimal value of $\lambda$ is still an intermediate value between 0 and 1. This plot matches Sutton's Figure 5. Optimal $\alpha$ values for each lambda were obtained by iterating through each value of $\alpha$ from 0.05 to 0.55 to find the value that minimized error for that particular value of $\lambda$. With this result, we affirm Sutton's assertion that $\lambda = 0$ is not optimal in this case because of the latency in propogating values between states.
