# Q-Learning
# http://mnemstudio.org/path-finding-q-learning-tutorial.htm

import json

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers

num_episodes = 1000


def run_episode(env, Q, learning_rate, discount, episode, render=False):
    observation = env.reset()
    done = False
    t_reward = 0
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    for i in range(max_steps):
        if done:
            break

        if render:
            env.render()

        curr_state = observation

        action = np.argmax(Q[curr_state, :] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))

        observation, reward, done, info = env.step(action)

        t_reward += reward

        # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        Q[curr_state, action] += learning_rate * (reward + discount * np.max(Q[observation, :]) - Q[curr_state, action])
        q_list.append(Q.tolist())
        observation_list.append(observation)

    return Q, t_reward


def train():
    from gym.envs.registration import register
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78,  # optimum = .8196
    )
    # env = gym.make('FrozenLake-v0')
    env = gym.make('FrozenLakeNotSlippery-v0')
    env = wrappers.Monitor(env, '/tmp/FrozenLake-experiment-6', force=True)
    learning_rate = 0.81
    discount = 0.96

    reward_per_ep = list()
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for i in range(num_episodes):
        Q, reward = run_episode(env, Q, learning_rate, discount, i, False)
        reward_per_ep.append(reward)
    # print "----------Next Episode---------"
    # print i
    plt.plot(reward_per_ep)

    return Q


if __name__ == '__main__':
    q_list = []
    observation_list = []
    q = train()
    print(q)

    f = open('data.json', 'w')
    f.write(json.dumps({'q_list': q_list, 'ob_list': observation_list}))
    f.close()
    # print(json.dumps(observation_list))
