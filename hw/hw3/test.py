import numpy as np
import json
import sys


def build_mdp(mdp_dict):
    gamma = mdp_dict.get('gamma')
    num_states = len(mdp_dict.get('states'))
    num_actions = len(set([
        a.get('id') for
        s in mdp_dict.get('states', []) for
        a in s.get('actions', [])
    ]))
    print num_states, num_actions

    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))
    print P
    print R

    for state in mdp_dict.get('states', []):
        state_id = state.get('id')
        for action in state.get('actions', []):
            action_id = action.get('id')
            for transition in action.get('transitions', []):
                next_state = transition.get('to')
                P[action_id, state_id, next_state] = transition.get('probability')
                R[next_state, action_id] = transition.get('reward')
    return gamma, P, R



with open(sys.argv[1]) as file_:
    mdp_dict = json.load(file_)

gamma, P, R = build_mdp(mdp_dict)
