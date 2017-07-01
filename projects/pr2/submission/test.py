import logging
logging.basicConfig(filename='log', level=logging.INFO, format='%(message)s', datefmt='')
import os
import numpy as np
from collections import deque
import gym
import sys
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = 'final-model.h5'
render = True

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


def huber_loss(x, y, delta=1):
    return K.mean((delta ** 2) * K.sqrt(1 + K.square((x - y) / delta)) - 1)


lr = 0.001
hidden_nodes = [400]
optimizer = Adam(lr=lr)
loss = huber_loss
activations = [LeakyReLU() for _ in xrange(len(hidden_nodes))]


def build_nn():
    nn = Sequential()
    nn.add(Dense(hidden_nodes[0], input_shape=env.observation_space.shape))
    nn.add(activations[0])
    for index in xrange(1, len(hidden_nodes)):
        nn.add(Dense(hidden_nodes[index]))
        nn.add(activations[index])
    nn.add(Dense(env.action_space.n, activation='linear'))
    return nn




normalization = {
    'mean': [0.07, 0.68, 0.08, -0.477, 0.00958, -0.1185, 0.0229, 0.00746],
    'std': [0.2419, 0.3268, 0.44679, 0.526, 0.6948, 0.67, 0.14958, 0.08605],
}

def normalize(vector):
        return (vector - normalization['mean']) / normalization['std']


model = build_nn()
model.compile(loss=loss, optimizer=optimizer)
model.load_weights(model_path)
rewards_history = deque(maxlen=100)
avg_rewards = 0
episode_num = 0
total_steps = 0

while episode_num < 100:
    episode_num += 1
    state = normalize(env.reset())
    episode_reward = 0
    done = False

    while not done:
        if render:
            env.render()
        Q_s = model.predict(state.reshape(1, 8))[0]
        action = np.argmax(Q_s)
        state_prime, reward, done, _ = env.step(action)
        state_prime = normalize(state_prime)
        total_steps += 1
        episode_reward += reward

        state = state_prime

    rewards_history.append(episode_reward)
    avg_rewards = np.mean(rewards_history)

    logging.info('%s', (
        episode_num,
        int(episode_reward),
        int(avg_rewards),
    ))
