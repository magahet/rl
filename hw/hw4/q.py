import numpy as np
from collections import deque
import gym
import time


alpha = 0.3
gamma = 0.9
epsilon = 0.8
duration = 60


env = gym.make('Taxi-v2')
Q = np.zeros((env.observation_space.n, env.action_space.n))
Q_last = Q.copy() + 1
rewards = deque()
episode_num = 0
avg_rewards = 0
start = time.time()

test_values = np.array([
    -11.374402515,
    4.348907,
    -0.5856821173,
    9.683,
    -12.8232660372
])

while True:
    episode_num += 1
    state = env.reset()
    steps = 0
    cum_rewards = 0

    while True:
        steps += 1

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        state_prime, reward, done, _ = env.step(action)
        cum_rewards += reward

        if reward == 20:
            Q[state, action] += alpha * (reward + 0 - Q[state, action])
            break

        Q[state, action] += alpha * (reward + gamma * np.max(Q[state_prime]) - Q[state, action])

        state = state_prime

    rewards.append(cum_rewards)

    if len(rewards) > 100:
        rewards.popleft()
        avg_rewards = sum(rewards) / 100.0
        if avg_rewards > 9.7:
            break

    values = np.array([
        Q[462, 4],
        Q[398, 3],
        Q[253, 0],
        Q[377, 1],
        Q[83, 5]
    ])
    delta_values_norm = np.linalg.norm(values - test_values)

    if episode_num % 10000 == 0:
        print episode_num, delta_values_norm 

    if delta_values_norm < 1e-10:
        break
