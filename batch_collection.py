import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import itertools
from utils import device
from model import ContinuousPolicy


def batch_collection(env, policy, seed, *, total_trajectories=16, smoothing=False):
    """Collect `total_trajectories` episodes from a SyncVectorEnv."""
    trajectories = []
    #TODO: implement batch_collection
    return trajectories

def pair_trajectories(trajs, temp=1.0, seed=None):
    rng = np.random.default_rng(seed)
    returns = np.array([np.sum(t[1]) for t in trajs])

    idx_pairs = list(zip(rng.permutation(len(trajs)), rng.permutation(len(trajs))))
    idx_pairs = [pair for pair in idx_pairs if pair[0] != pair[1]]

    pair_data = []
    #TODO: implement pair_trajectories
    return pair_data

def collect_pair_data(policy, seed, total_trajectories=16, smoothing=False):
    env_fns = [lambda: gym.make("Swimmer-v5") for _ in range(16)]
    env = SyncVectorEnv(env_fns)

    example_env = gym.make("Swimmer-v5")
    obs_dim = example_env.observation_space.shape[0]
    act_dim = example_env.action_space.shape[0]
    example_env.close()

    trajectories = batch_collection(env, policy, seed, total_trajectories=total_trajectories, smoothing=smoothing)
    return pair_trajectories(trajectories, seed=seed)

def main():
    num_envs = 32 # set this based on your hardware
    seed = 42

    env_fns = [lambda: gym.make("Swimmer-v5") for _ in range(num_envs)]
    env = SyncVectorEnv(env_fns)

    example_env = gym.make("Swimmer-v5")
    obs_dim = example_env.observation_space.shape[0]
    act_dim = example_env.action_space.shape[0]
    example_env.close()

    policy = ContinuousPolicy(obs_dim, act_dim)
    policy.load_state_dict(torch.load("swimmer_checkpoint.pt", weights_only=False))

    trajectories = batch_collection(env, policy, seed, total_trajectories=5000)

    mean_reward = np.mean([np.sum(traj[1]) for traj in trajectories])
    std_reward = np.std([np.sum(traj[1]) for traj in trajectories])
    print(f"Mean reward of trajectories: {mean_reward}, std: {std_reward}")
    pair_data = pair_trajectories(trajectories, seed=seed)
    torch.save(pair_data, "pair_data.pt")

if __name__ == "__main__":
    main()
