import os
import numpy as np
from collections import deque
import gym
import sys
import random
from keras.models import (Model, Sequential)
from keras.layers import (Input, Dense, merge)
from keras.optimizers import Adam
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 16


duel = True
double = True
priority_replay = True

per_alpha = 0.6
per_epsilon = 0.01

hidden_nodes = [50, 40]
optimizer = Adam(lr=0.0005)
activation = 'relu'

gamma = 0.99
epsilon_end = 100
epsilon_min = 0.01
training_freq = 1
target_update_freq = 600
batch_size = 32
history_size = 5e5

epsilon_delta = (1 - epsilon_min) / epsilon_end
np.random.seed(seed)
random.seed(seed)
env = gym.make('LunarLander-v2')
env.seed(seed)
state_sum = np.zeros(8)
statesqr_sum = np.zeros(8)
normalization = {
    'mean': [0.07, 0.68, 0.08, -0.477, 0.00958, -0.1185, 0.0229, 0.00746],
    'std': [0.2419, 0.3268, 0.44679, 0.526, 0.6948, 0.67, 0.14958, 0.08605],
}


def hubert_loss(x, y):
    return K.mean(K.sqrt(1 + K.square(x - y)), axis=-1)


def build_nn():
    if duel:
        input_ = Input(shape=env.observation_space.shape)
        h_a = Dense(hidden_nodes[0], activation=activation)(input_)
        for node_count in hidden_nodes[1:]:
            h_a = Dense(node_count, activation=activation)(h_a)
        advantage = Dense(env.action_space.n)(h_a)

        h_v = Dense(hidden_nodes[0], activation=activation)(input_)
        for node_count in hidden_nodes[1:]:
            h_v = Dense(node_count, activation=activation)(h_v)
        value = Dense(1)(h_v)

        policy = merge([advantage, value],
                       mode=lambda x: x[0]-K.mean(x[0])+x[1],
                       output_shape=(env.action_space.n,))

        return Model(inputs=input_,  outputs=policy)
    else:
        nn = Sequential()
        nn.add(Dense(hidden_nodes[0], activation=activation,
                     input_shape=env.observation_space.shape))
        for node_count in hidden_nodes[1:]:
            nn.add(Dense(node_count, activation=activation))
        nn.add(Dense(env.action_space.n, activation='linear'))
        return nn


def normalize(vector):
    return (vector - normalization['mean']) / normalization['std']


if len(sys.argv) > 1:
    save_path = sys.argv[1]
    if os.isfile(save_path):
        from keras.models import load_model
        nn = load_model(save_path)
else:
    save_path = None
    Q_nets = [build_nn(), build_nn()]
    for i in xrange(2):
        Q_nets[i].compile(loss=hubert_loss, optimizer=optimizer)


def learn():
    sample_weights = np.power(priority, per_alpha)
    sample_weights /= np.sum(sample_weights)

    x, y = [], []
    if priority_replay:
        samples = np.random.choice(len(history),
                                   size=min(len(history), batch_size),
                                   replace=False, p=sample_weights)
    else:
        samples = np.random.choice(len(history),
                                   size=min(len(history), batch_size),
                                   replace=False)

    for index in samples:
        state, action, state_prime, reward, done = history[index]
        Q_s = Q_nets[0].predict(state.reshape(1, 8))[0]
        Q_s_prime = Q_nets[1].predict(state_prime.reshape(1, 8))[0]

        if done:
            Q_s[action] = reward
        else:
            td_error = reward + gamma * np.max(Q_s_prime) - Q_s[action]
            priority[index] = max(abs(td_error), per_epsilon)
            Q_s[action] = reward + gamma * np.max(Q_s_prime)

        x.append(state.flatten())
        y.append(Q_s.flatten())
    output = Q_nets[0].fit(np.array(x), np.array(y), batch_size=batch_size,
                           epochs=1, verbose=False)
    return output.history['loss'][-1]


history = deque(maxlen=history_size)
priority = deque(maxlen=history_size)
rewards_history = deque(maxlen=100)
epsilon = 1.0
episode_num = 0
avg_rewards = 0
total_steps = 0


while avg_rewards < 200:
    episode_num += 1
    state = env.reset()
    episode_reward = 0
    done = False
    episode_loss = 0

    while not done:
        priority_max = np.max(priority) if priority else per_epsilon

        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            Q_s = Q_nets[0].predict(state.reshape(1, 8))[0]
            action = np.argmax(Q_s)

        if episode_num % 10 == 0:
            env.render()

        state_prime, reward, done, _ = env.step(action)
        total_steps += 1
        episode_reward += reward

        history.append((state, action, state_prime, reward, done))
        priority.append(priority_max)
        state = state_prime

        if total_steps % training_freq == 0:
            episode_loss += learn()

        if total_steps % target_update_freq == 0:
            Q_nets[1].set_weights(Q_nets[0].get_weights())

    rewards_history.append(episode_reward)
    avg_rewards = np.mean(rewards_history)
    epsilon = max(epsilon - epsilon_delta, epsilon_min)

    print (episode_num, int(episode_reward), int(avg_rewards),
           epsilon, int(episode_loss))


Q_nets[0].save('/tmp/final-model')
