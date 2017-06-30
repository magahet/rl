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

Q value function approximation works by calculating $Q(s, a) = reward + \gamma argmax(Q(s', a))$ from experience and submitting (s, Qs) pairs to the neural network for training. Q(s', a) values are obtained from the neural network, as well as argmax(Q(s, a)) values for choosing agent actions.

### Neural Network
DQN (Deep Q Network) refers to a Q-learning agent with a neural network for learning and predicting Q values at given (state, action) pairs. The 'Deep" part refers to a network with many hidden layers (including convolutions), as was used in the original Deep Mind research. The agent used to solve LunarLander uses a single hidden dense layer with 400 nodes. This was enough to approximate Q values for this environment without requiring longer training times on a more complex network. This agent could then be considered a simple QN.

**Layers** --
I tested many combinations of layers and nodes to find the simplest network capable of approximating Q values with enough granularity to guide the lander towards the goal. I tested single hidden layers with nodes between 16 and 1024 and two hidden layer networks with the second layer either equal or half that of the fist layer. Each of these experiments resulted in worse or similar initial performance or increased training time.

**Activation and Loss** --
Neural network nodes require an activation function to map inputs to outputs. A common function used in DQN is ReLU (x for x $\geq$ 0 else 0) in hidden nodes. I encountered a problem with weights no longer updating in certain nodes (dead nodes), and found that LeakyReLU (x for x $\geq$ 0 else 0.01x) allowed the network to recover in those situations. The loss function for Q networks are usually MSE, but I noticed occasionally high errors depending on changing terrain. To make the agent robust to these outliers, I used a pseudo-huber loss function to reduce the effect of high error terms.

### Updates
I tested a variety of update strategies, all of them included experience replay. This is process of storing (s, a, s', r, done) tuples and running model training on batches from memory. This prevents the model from learning from just the most recent (highly correlated) experiences. Memory size was adjusted as a hyper-parameter and finally set at 5e5. Batch sizes where adjusted from 16 to 512 (final at 256). Batch sizes were changed based on training frequency. Testing was done on training after a set number of episodes (1 to 100) and set number of steps (1 to 1000). The right balance seemed to be to run a batch of 256 every 500 steps.

### Epsilon Strategy
Agent actions were chosen based on Q values with a probability of $1 - \epsilon$ and randomly chosen (uniform) otherwise. $\epsilon$ was then adjusted based on different strategies. I tested decay strategies including geometric, linear, tiered linear (min values based on current avg reward), VDBE (Adaptive epsilon-greedy strategy based on value differences), and Contextual-Epsilon-greedy (based on current avg reward). The adaptive and contextual decay strategies were abandoned as they added additional complexity without significant performance improvements. A linear strategy with $\Delta \epsilon = 0.001$ as a hyper-parameter was used for the final agent.


### DQN Improvements
I attempted a number of modified Q network strategies such as a separate network for stabilizing target values (double DQN); separate networks for values and action advantages (Dueling DQN), and prioritizing experience replay based on TD error (PER). These were all eventually abandoned either due to increased training runtimes, or insignificant performance improvements. Unfortunately, there isn't enough space to cover each of these learning experiences individually or in any depth. Examples of these implementations are available in the code repository under the improvements directory.


## Results

![Training Rewards\label{training}](fig/training.png){#id .class width=75%}

Figure \ref{training} shows the results of training the agent until average reward reaches 200.
