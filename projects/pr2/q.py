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
gamma = 0.99
epsilon = 0.6
epsilon_decay = 0.997
duration = 60
num_nodes = 32


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

history = deque(maxlen=1e3)
rewards_history = deque(maxlen=100)
episode_num = 0
avg_rewards = 0
start = time.time()
np.random.seed(42)

while avg_rewards < 200:
    episode_num += 1
    # alpha = 1.0 / episode_num
    state = env.reset()
    episode_reward = 0
    done = False
    steps = 0

    while not done:
        steps += 1
        if steps >= 5000:
            break
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            Q_s = nn.predict(state.reshape(1, 8))[0]
            action = np.argmax(Q_s)

        if episode_num % 10 == 0:
            env.render()
        state_prime, reward, done, _ = env.step(action)
        history.append((state, action, state_prime, reward, done))
        episode_reward += reward
        state = state_prime

    rewards_history.append(episode_reward)
    avg_rewards = np.mean(rewards_history)

    epsilon = max(epsilon * epsilon_decay, 0.01)

    #if time.time() - start > 5:
    if episode_num % 10 == 0:
        start = time.time()
        print episode_num, avg_rewards, epsilon

    if episode_num % 10 != 0:
        continue

    for _ in xrange(64):
        state, action, state_prime, reward, done = (
            history[np.random.choice(len(history))])
        Q_s = nn.predict(state.reshape(1, 8))[0]
        Q_s_prime = nn.predict(state_prime.reshape(1, 8))[0]

        if done:
            Q_s[action] = reward
        else:
            Q_s[action] = reward + gamma * np.max(Q_s_prime)

        nn.fit(state.reshape(1, 8), Q_s.reshape(1, 4), epochs=1, verbose=False)

    if save_path:
        nn.save(save_path)
