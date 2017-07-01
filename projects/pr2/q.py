import logging
logging.basicConfig(filename='log', level=logging.INFO, format='%(message)s', datefmt='')
import os
import numpy as np
from collections import deque
import gym
import sys
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
env = gym.make('LunarLander-v2')
env.seed(seed)

from keras.models import (Model, Sequential)
from keras.layers import (Input, Dense, merge)
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU


def hubert_loss(x, y):
    return K.mean(K.sqrt(1 + K.square(x - y)), axis=-1)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# hyper-parameters
lr = 0.001
hidden_nodes = [400]
optimizer = Adam(lr=lr)
loss = hubert_loss
activations = [LeakyReLU() for _ in xrange(len(hidden_nodes))]
gamma = 0.99
epsilon_delta = 0.001
training_freq = 500
batch_size = 256
history_size = 1e6
history_store_freq = 1
history_min = 4 * batch_size

normalization = {
    'mean': [0.07, 0.68, 0.08, -0.477, 0.00958, -0.1185, 0.0229, 0.00746],
    'std': [0.2419, 0.3268, 0.44679, 0.526, 0.6948, 0.67, 0.14958, 0.08605],
}


def normalize(vector):
        return (vector - normalization['mean']) / normalization['std']


def build_nn():
    nn = Sequential()
    nn.add(Dense(hidden_nodes[0], input_shape=env.observation_space.shape))
    nn.add(activations[0])
    for index in xrange(1, len(hidden_nodes)):
        nn.add(Dense(hidden_nodes[index]))
        nn.add(activations[index])
    nn.add(Dense(env.action_space.n, activation='linear'))
    return nn


def learn():
    x, y = [], []
    for sample in random.sample(history, min(len(history), batch_size)):
        state, action, state_prime, reward, done = sample
        Q_s = model.predict(state.reshape(1, 8))[0]
        Q_s_prime = model.predict(state_prime.reshape(1, 8))[0]

        if done:
            Q_s[action] = reward
        else:
            Q_s[action] = reward + gamma * np.max(Q_s_prime)

        x.append(state.flatten())
        y.append(Q_s.flatten())
    output = model.fit(np.array(x), np.array(y), verbose=False)
    model.save('model.h5')
    return output.history['loss'][-1]


model = build_nn()
model.compile(loss=loss, optimizer=optimizer)


history = deque(maxlen=history_size)
rewards_history = deque(maxlen=100)
epsilon = 1.0
episode_num = 0
avg_rewards = 0
total_steps = 0


while avg_rewards < 200:
    episode_num += 1
    state = normalize(env.reset())
    episode_reward = 0
    episode_shaped_reward = 0
    done = False
    episode_loss = 0

    while not done:
        env.render()
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            Q_s = model.predict(state.reshape(1, 8))[0]
            action = np.argmax(Q_s)

        state_prime, reward, done, _ = env.step(action)
        state_prime = normalize(state_prime)
        state_prime = state_prime
        total_steps += 1
        episode_reward += reward

        if total_steps % history_store_freq == 0:
            history.append((state, action, state_prime, reward, done))

        state = state_prime

        if total_steps % training_freq == 0 and len(history) >= history_min:
            episode_loss += learn()

    epsilon = max(epsilon - epsilon_delta, 0.05)

    rewards_history.append(episode_reward)
    avg_rewards = np.mean(rewards_history)

    logging.info('episode: %d, reward: %d, avg_reward: %d, epsilon: %.2f',
        episode_num, int(episode_reward), int(avg_rewards), epsilon)

model.save('final-model.h5')
