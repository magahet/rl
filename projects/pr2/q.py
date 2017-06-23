import os
import numpy as np
from collections import deque
import gym
import time
import sys
from keras.models import Sequential
from keras.layers import Dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


alpha = 1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
duration = 60
num_nodes = 48


env = gym.make('LunarLander-v2')

if len(sys.argv) > 1:
    save_path = sys.argv[1]
    if os.isfile(save_path):
        from keras.models import load_model
        nn = load_model(save_path)
else:
    save_path = None
    nn = Sequential()
    nn.add(Dense(num_nodes, input_shape=env.observation_space.shape,
                 activation='relu'))
    nn.add(Dense(num_nodes, activation='relu'))
    nn.add(Dense(env.action_space.n, activation='linear'))
    nn.compile(loss='mse', optimizer='adam')

history = deque()
rewards_history = deque()
episode_num = 0
avg_rewards = 0
start = time.time()
np.random.seed(42)

while avg_rewards < 200:
    episode_num += 1
    epsilon = 0.0
    # alpha = 1.0 / episode_num
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            Q_s = nn.predict(state.reshape(1, 8))[0]
            action = np.argmax(Q_s)

        state_prime, reward, done, _ = env.step(action)
        history.append((state, action, state_prime, reward, done))
        if len(history) > 1e6:
            history.popleft()
        episode_reward += reward
        state = state_prime

    epsilon = min(1 - reward / 200, 0.8)

    rewards_history.append(episode_reward)
    if len(rewards_history) > 100:
        rewards_history.popleft()
    avg_rewards = np.mean(rewards_history)

    if time.time() - start > 5:
        start = time.time()
        print episode_num, avg_rewards, epsilon

    if episode_num % 100 != 0:
        continue

    for _ in xrange(100):
        state, action, state_prime, reward, done = (
            history[np.random.choice(len(history))])
        Q_s = nn.predict(state.reshape(1, 8))[0]
        Q_s_prime = nn.predict(state_prime.reshape(1, 8))[0]

        if done:
            Q_s[action] = reward
        else:
            Q_s[action] = reward + gamma * np.max(Q_s_prime)

        nn.fit(state.reshape(1, 8), Q_s.reshape(1, 4), verbose=False)

    if save_path:
        nn.save(save_path)
