import random
from game import Soccer


env = Soccer()
state, rewards, done = env.reset()

while not done:
    actions = random.sample(range(5), 2)
    next_state, rewards, done = env.step(actions)
    print state, actions, next_state, rewards, done
    state = next_state
