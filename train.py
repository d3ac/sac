import os
import sys
if sys.platform == 'win32':
    sys.path.append(os.path.expanduser('C:/Users/10485/Desktop/科研训练/uavenv'))
else:
    sys.path.append(os.path.expanduser('~/Desktop/科研训练/uav env'))
from UAVenv.uav.uav import systemEnv
os.environ['PARL_BACKEND'] = 'torch'
import warnings

import gym
import argparse
import numpy as np
from parl.utils import logger, summary, ReplayMemory
from parl.env import ActionMappingWrapper, CompatWrapper
from uav_model import uavModel
from uav_agent import Agent
from algorithm import SAC
import pandas as pd

WARMUP_STEPS = 1e2
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

# Run episode for training
def run_train_episode(agent, env, rpm):
    obs = env.reset()
    episode_reward = 0
    episode_steps = 0
    while True:
        episode_steps += 1
        # Select action randomly or according to policy
        action = agent.sample(obs)
        # Perform action
        next_obs, reward, done, _ = env.step(action)
        for i in range(env.n_clusters):
            rpm[i].append(obs[i], action[i], reward[i], next_obs[i], done[i])
        # Store data in replay memory
        obs = next_obs
        episode_reward += reward
        # Train agent after collecting sufficient data
        if rpm[0].size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = [], [], [], [], []
            for i in range(env.n_clusters):
                _obs, _act, _rew, _next_obs, _done = rpm[i].sample_batch(BATCH_SIZE)
                
                batch_obs.append(_obs)
                batch_action.append(_act)
                batch_reward.append(_rew)
                batch_next_obs.append(_next_obs)
                batch_done.append(_done)

            batch_obs = np.array(batch_obs)
            batch_action = np.array(batch_action)
            batch_reward = np.array(batch_reward)
            batch_next_obs = np.array(batch_next_obs)
            batch_done = np.array(batch_done)

            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
        if done.all():
            break
    return np.mean(episode_reward), np.mean(episode_steps)


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            if done.all():
                break
    avg_reward /= eval_episodes
    return np.mean(avg_reward)


def main():
    logger.info("------------------- SAC ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")

    env = CompatWrapper(systemEnv())
    env.seed(args.seed)
    obs_dim = env.observation_space[0].shape
    action_dim = env.action_space[0].nvec
    n_clusters = env.n_clusters

    # Initialize model, algorithm, agent, replay_memory
    model = uavModel(obs_dim, action_dim, n_clusters, BATCH_SIZE)
    algorithm = SAC(model, gamma=GAMMA, tau=TAU, alpha=args.alpha, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm)
    rpm = [ReplayMemory(MEMORY_SIZE, obs_dim[0], len(action_dim)) for _ in range(env.n_clusters)]

    total_steps = 0
    test_flag = 0
    data = []
    while total_steps < args.train_total_steps:
        # Train episode
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        total_steps += episode_steps
        logger.info('Total Steps: {} Reward: {}'.format(total_steps, episode_reward))

        # Evaluate episode
        avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES)
        logger.info('Evaluation over: {} episodes, Reward: {}'.format(EVAL_EPISODES, avg_reward))
        data.append(avg_reward)
        Temp = pd.DataFrame(data)
        Temp.to_csv('data.csv', index=False, header=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="uav-v0", help='Mujoco gym environment name')
    parser.add_argument("--seed", default=0, type=int, help='Sets Gym seed')
    parser.add_argument("--train_total_steps", default=3e6, type=int, help='Max time steps to run environment')
    parser.add_argument('--test_every_steps', type=int, default=int(5e3), help='The step interval between two consecutive evaluations')
    parser.add_argument("--alpha", default=0.2, type=float, help='Determines the relative importance of entropy term against the reward')
    args = parser.parse_args()
    main()