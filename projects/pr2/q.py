import os
import numpy as np
from collections import deque
import gym
import sys
import random
from keras.models import (Model, Sequential)
from keras.layers import (Input, Dense, merge, BatchNormalization)
from keras.optimizers import (Adam, SGD)
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 16


duel = False
double = True
priority_replay = False

per_alpha = 0.6
per_epsilon = 0.01

hidden_nodes = [50, 40]
optimizer = Adam(lr=0.0005)
activation = 'relu'
loss = lambda x, y: K.mean(K.sqrt(1 + K.square(x - y)), axis=-1)

gamma = 0.99
epsilon_end = 100
epsilon_min = 0.0
epsilon_decay = 0.975
training_freq = 1
target_update_freq = 600
batch_size = 32
history_size = 5e5
history_min = 1e3

epsilon_delta = (1 - epsilon_min) / epsilon_end
np.random.seed(seed)
random.seed(seed)
env = gym.make('LunarLander-v2')
env.seed(seed)
state_sum = np.zeros(8)
statesqr_sum = np.zeros(8)
normalization = {
    'mean': [0.0704982, 0.68328412, 0.08038708, -0.47703002, 0.00958349, -0.11847555, 0.02290007, 0.00746066],
    'std': [0.24193065, 0.32684624, 0.44679473, 0.52634632, 0.69483146, 0.67285418, 0.14958495, 0.08605228],
}


def hubert_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )


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

        policy = merge([advantage, value], mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (env.action_space.n,))

        return Model(input=input_,  output=policy)
    else:
        nn = Sequential()
        nn.add(Dense(hidden_nodes[0], activation=activation, input_shape=env.observation_space.shape))
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
        Q_nets[i].compile(loss=loss, optimizer=optimizer)


history = deque(maxlen=history_size)
priority = deque(maxlen=history_size)
rewards_history = deque(maxlen=100)
epsilon = 1.0
episode_num = 0
avg_rewards = 0
steps_since_update = 0
render = False
m = 0
good = False
total_steps = 0

while avg_rewards < 200:
    episode_num += 1
    # alpha = 1.0 / episode_num
    state = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    priority_max = np.max(priority) if priority else per_epsilon
    priority_mean = np.mean(priority) if priority else per_epsilon

    while not done and steps < 500:
        steps += 1
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space.n)
            # action = np.random.choice(env.action_space.n, p=(0.4, 0.2, 0.2, 0.2))
        else:
            Q_s = Q_nets[0].predict(np.array(normalize(state)))[0]
            action = np.argmax(Q_s)

        if (render and episode_num % 10 == 0) or episode_num % 50 == 0:
            env.render()
            if good:
                print 'OK'

        state_prime, reward, done, _ = env.step(action)
        state_sum += state_prime
        statesqr_sum += (state_prime ** 2)
        total_steps += 1
        episode_reward += reward

        x, y, vx, vy, angle, v_angle, left_leg, right_leg = state_prime
        # if steps == 500 and not done:
        #     reward -= 1000
        # if all([abs(x) < 0.2, abs(vx) < 0.2, abs(angle) < 0.1, abs(v_angle) < 0.1, vy > -0.3, vy < -0.2]):
        #     good = True
        #     reward += 1
        # else:
        #     good = False
        # if action in (1, 3):
        #    reward -= 0.3
        # reward -= 4 ** abs(state[2])
        # reward -= 4 ** abs(state[4])
        # reward -= 4 ** abs(state[5])
        history.append((state, action, state_prime, reward, done))
        priority.append(priority_max)
        state = state_prime

    if not render:
        render = (avg_rewards > -130 and episode_num > 100) or episode_num >= 799

    rewards_history.append(episode_reward)
    avg_rewards = np.mean(rewards_history)

    epsilon = max(epsilon * epsilon_decay, 0.001)
    # epsilon = max(epsilon - epsilon_delta, epsilon_min)
    # if avg_rewards < -100:
    #     epsilon = max(epsilon, 0.3)
    # elif avg_rewards < 0:
    #     epsilon = max(epsilon, 0.2)
    # elif avg_rewards >= 100:
    #     epsilon = 0.1


    if episode_num % 10 == 0:
        print episode_num, int(episode_reward), int(avg_rewards), epsilon, loss
        mean = state_sum / float(total_steps)
        sqrmean = statesqr_sum / float(total_steps)
        # print mean, np.sqrt(sqrmean - (mean **2)) 

    if len(history) < history_min or episode_num % training_freq != 0:
        continue

    sample_weights  = np.power(priority, per_alpha)
    sample_weights /= np.sum(sample_weights)

    x, y = [], []
    if priority_replay:
        samples = np.random.choice(len(history), size=batch_size, replace=False, p=sample_weights)
    else:
        samples = np.random.choice(len(history), size=batch_size, replace=False)

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
    output = Q_nets[0].fit(np.array(x), np.array(y), batch_size=batch_size, epochs=1, verbose=False)
    loss = output.history['loss'][-1]

    if steps_since_update > target_update_freq:
        steps_since_update = 0
        Q_nets[1].set_weights(Q_nets[0].get_weights())


Q_nets[0].save('/tmp/final-model')
