#!/usr/bin/env python


import os
import json
import logging as log
import argparse
import numpy as np


class Env:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.P = np.zeros((self.nS, self.nA), dtype=np.ndarray)

    def __repr__(self):
        return 'number of states: {}, max number of action: {}'.format(self.nS,
                                                                       self.nA)


def policy_evaluation(policy, env, discount_factor=.75, theta=0.000001):
    """
    Adapted from Denny Britz repo
    """
    V = np.zeros(env.nS)
    # print('evaluating policy')
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (
                        reward + discount_factor * V[next_state] * (not done)
                    )
                    # print(s, a, prob, next_state, reward, done, v)
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            # print(delta)
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_evaluation,
                       discount_factor=0.75):
    """
    Adapted from Denny Britz repo
    """
    # Start with a random policy
    policy = np.random.random([env.nS, env.nA])

    iterations = 0
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        iterations += 1
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (
                        reward + discount_factor * V[next_state]
                    )
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V, iterations


def json_mdp_to_openai_env(mdp):

    log.info('inside json_mdp_to_openai_env')

    # for  prob, next_state, reward, done in env.P[s][a]:
    nstates = len(mdp['states'])
    max_nactions = 0
    for state in mdp['states']:
        if len(state['actions']) > max_nactions:
            max_nactions = len(state['actions'])

    log.info('initializing MDP with nstates {} and nactions {}'.format(
        nstates, max_nactions
    ))
    env = Env(nstates, max_nactions)

    for state in mdp['states']:
        for action in state['actions']:
            trans = []
            for transition in action['transitions']:
                trans.append((transition['probability'], transition['to'],
                              transition['reward'],
                              # """
                              True if state['id'] == transition['to']
                              # and nactions == 1
                              and transition['probability'] == 1.0
                              and transition['reward'] == 0.0 else False))
                # """
                # False))
            env.P[state['id']][action['id']] = trans

    for si in range(nstates):
        for ai in range(max_nactions):
            log.info('for state, action pair ({}, {}) = {}'.format(
                si, ai, env.P[si][ai]
            ))

    log.info(env)
    log.info('mdp transformed into OpenAI format')
    return env


def get_iterations(mdp):

    env = json_mdp_to_openai_env(mdp)

    log.info('running policy improvement')
    # (policy, v, it)
    results = np.array([(policy_improvement(env)) for _ in range(100)])
    log.info("Policy Probability Distribution:")
    log.info(results[0, 0])
    log.info("")

    log.info("Value Function:")
    log.info(results[0, 1])
    log.info("")

    log.info("Number Iterations:")
    log.info(results[:, 2])
    log.info('minimum')
    log.info(np.min(results[:, 2]))
    log.info('maximum')
    log.info(np.max(results[:, 2]))
    log.info('mean')
    log.info(np.mean(results[:, 2]))
    log.info('median')
    log.info(np.median(results[:, 2]))
    log.info("")
    return results[:, 2].max()


def verify_mdp(mdp):

    log.info('in the verify_mdp function')
    nstates = len(mdp['states'])
    if nstates > 30:
        log.critical('too many states: (' + str(nstates) + ')! seriously?')
        raise Exception('too many states')
    log.debug('legal number of states')

    states = []
    fixed_n_actions = len(mdp['states'][0]['actions'])
    for s in mdp['states']:
        log.debug('\nstate id ' + str(s['id']))

        nactions = len(s['actions'])
        if nactions > 2:
            log.critical('too many actions on a single state: (' +
                         str(nactions) + ')! won\'t do it!')
            raise Exception('one of the states has more than 2 actions')
        if fixed_n_actions != nactions:
            log.critical(
                'states should have the same number of actions. Found: (' +
                str(nactions) + ') and (' + str(fixed_n_actions) +
                ') clean that up!'
            )
            raise Exception('one of the states has more than 2 actions')

        log.debug('state has correct number of actions')

        actions = []
        for a in s['actions']:
            log.debug('\taction id ' + str(a['id']))

            prob = 0
            trans = []
            for t in a['transitions']:
                log.debug('\t\ttransition id ' + str(t['id']))

                if not t['probability']:
                    log.critical('transition with zero probability, '
                                 'why would you add that???')
                    raise Exception(
                        'no probability found on transition, cannot add it'
                    )

                if t['probability'] < 0:
                    log.critical('negative probability, '
                                 'what am I supposed to do with that???')
                    raise Exception('probability is negative, cannot add it')

                if t['probability'] > 1:
                    log.critical('a probability greater than 1?? '
                                 'you should go to Vegas!')
                    raise Exception(
                        'probability is greater than 1, cannot add it'
                    )

                prob += t['probability']
                trans.append(t)

            if prob != 1:
                log.critical('transition probabilities do not equal 1 '
                             'for a single action, something\'s wrong...')
                raise Exception('the sum of the probabilities must equal 1 '
                                'for all transitions of a single action')
            a['transitions'] = trans
            actions.append(a)

        s['actions'] = actions
        states.append(s)

    mdp['states'] = states
    return mdp


def visualize_mdp(mdp, filename):
    import pydot
    import networkx as nx
    from networkx.drawing.nx_agraph import write_dot

    G = nx.DiGraph()

    for s in mdp['states']:
        for a in s['actions']:
            for t in a['transitions']:
                ecolor = 'red' if a['id'] else 'green'
                elabel = 'p={}, r={}'.format(t['probability'], t['reward'])
                G.add_edge(s['id'], t['to'],
                           color=ecolor,
                           label=elabel)

    write_dot(G, filename.replace('.json', '.dot'))
    g = pydot.graph_from_dot_file(filename.replace('.json', '.dot'))
    g.write_png(filename.replace('.json', '.png'))
    os.remove(filename.replace('.json', '.dot'))
    return filename.replace('.json', '.png')


def main(args):
    """
    """
    log.info('Verbose output enabled ' +
             str(log.getLogger().getEffectiveLevel()))
    log.debug(args)

    filename = args.mdp_path
    log.info('attempting to load MDP at ' + filename)
    with open(filename) as data_file:
        mdp = json.load(data_file)
    log.debug('file loaded successfully')

    log.info('verifying mdp')
    try:
        mdp = verify_mdp(mdp)
    except:
        log.fatal('MDP has problems. Cannot proceed!')
        exit(-1)

    if args.check_only:
        log.info('mdp was correct and checking only')
        exit(0)

    if args.print_iterations:
        niterations = get_iterations(mdp)
        log.info('mdp gave number of iterations ' + str(niterations))
        print('number of iterations: ' + str(niterations))

    if args.visualize_mdp:
        log.info('saving json visualization')
        png_path = visualize_mdp(mdp, filename)
        log.info('file found at ' + png_path)
    log.info('end of script')


if __name__ == '__main__':
    """
    Loads the script and parses the arguments
    """
    parser = argparse.ArgumentParser(
        description='Reinforcement Learning and Decision Making, HW4 Tester'
    )
    parser.add_argument(
        '-v',
        help='logging level set to ERROR',
        action='store_const', dest='loglevel', const=log.ERROR,
    )
    parser.add_argument(
        '-vv',
        help='logging level set to INFO',
        action='store_const', dest='loglevel', const=log.INFO,
    )
    parser.add_argument(
        '-vvv',
        help='logging level set to DEBUG',
        action='store_const', dest='loglevel', const=log.DEBUG,
    )

    # json path
    parser.add_argument(
        '-m', '--mdp',
        help='Path to the MDP json file',
        dest='mdp_path', type=str, required=True,
    )
    # verify mdp
    parser.add_argument(
        '-c', '--check_only',
        help='Flag to only check valid MDP on JSON file',
        dest='check_only', action='store_true',
    )
    # iterations
    parser.add_argument(
        '-i', '--iterations',
        help='Calculate how many iterations PI takes to solve this',
        dest='print_iterations', action='store_true',
    )
    # visualize
    parser.add_argument(
        '-s', '--visualize',
        help='Visualize MDP (export to png)',
        dest='visualize_mdp', action='store_true',
    )

    args = parser.parse_args()
    if args.loglevel:
        log.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=args.loglevel)
    else:
        log.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=log.CRITICAL)

    main(args)
