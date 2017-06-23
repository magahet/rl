import numpy as np
from collections import deque
import gym
import time
import os
import sys
from keras.models import Sequential 
from keras.layers import Dense


alpha = 1.0
gamma = 0.9
node_count = 128


env = gym.make('LunarLander-v2')

if len(sys.argv) > 1:
    save_path = sys.argv[1]
    if os.isfile(save_path):
        from keras.models import load_model
        nn = load_model(save_path)
else:
    save_path = None
    nn = Sequential()
    nn.add(Dense(node_count, input_shape=env.observation_space.shape, activation='relu'))
    nn.add(Dense(node_count, activation='relu'))
    nn.add(Dense(env.action_space.n, activation='linear'))
    nn.compile(loss='mean_squared_error', optimizer='adam')

history = deque()
rewards = deque()
episode_num = 0
avg_rewards = 0
start = time.time()

while True:
    episode_num += 1
    epsilon = 0.0
    # alpha = 1.0 / episode_num
    state = env.reset()
    cum_rewards = 0

    while True:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            Q_s = nn.predict(state.reshape(1, 8))[0]
            action = np.argmax(Q_s)

        state_prime, reward, done, _ = env.step(action)
        history.append((state, action, state_prime, reward, done))
        if len(history) > 1e6:
            history.popleft()
        cum_rewards += reward
        state = state_prime

        if done:
            break

    rewards.append(cum_rewards)
    if len(rewards) > 100:
        rewards.popleft()

    avg_rewards = np.mean(rewards)

    if avg_rewards >= 200:
        break

    if time.time() - start > 5:
        start = time.time()
        print episode_num, avg_rewards

    if episode_num % 200 != 0:
        continue

    x = []
    y = []
    for _ in xrange(100):
        state, action, state_prime, reward, done = history[np.random.choice(len(history))]
        Q_s = nn.predict(state.reshape(1, 8))[0]
        Q_s_prime = nn.predict(state_prime.reshape(1, 8))[0]

        if done:
            Q_s[action] += reward
        else:
            Q_s[action] += alpha * (reward + gamma * np.max(Q_s_prime))

        x.append(state)
        y.append(Q_s)

    nn.train_on_batch(np.array(x), np.array(y))


    if save_path:
        nn.save(save_path)

if not save_path:
    nn.save_path('/tmp/trained-model')
