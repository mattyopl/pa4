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
    policy.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    trajectories = []
    obs, _ = env.reset(seed=seed)

    curr_trajs = [[] for _ in range(env.num_envs)]  # hold obs
    curr_rewards = [[] for _ in range(env.num_envs)]  # hold rewards
    curr_actions = [[] for _ in range(env.num_envs)]  # hold actions
    curr_logps = [[] for _ in range(env.num_envs)]  # hold logps

    num_collected = 0

    with torch.no_grad():
        while num_collected < total_trajectories:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            result_sample_actions = policy.sample_action(obs_tensor, smooth=smoothing)
            actions = result_sample_actions[0].cpu().numpy()
            logps = result_sample_actions[1].cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = env.step(actions)

            for i in range(env.num_envs):
                curr_trajs[i].append(obs[i])
                curr_rewards[i].append(rewards[i])
                curr_actions[i].append(actions[i])
                curr_logps[i].append(logps[i])

                done = terminated[i] or truncated[i]
                if done:
                    trajectories.append((
                        np.array(curr_trajs[i])[:1000],
                        np.array(curr_rewards[i])[:1000],
                        np.array(curr_actions[i])[:1000],
                        np.array(curr_logps[i])[:1000]
                    ))
                    curr_trajs[i] = []
                    curr_rewards[i] = []
                    curr_actions[i] = []
                    curr_logps[i] = []
                    num_collected += 1
                    if num_collected >= total_trajectories:
                        break
            obs = next_obs

    return trajectories

def pair_trajectories(trajs, temp=1.0, seed=None):
    rng = np.random.default_rng(seed)
    returns = np.array([np.sum(t[1]) for t in trajs])

    idx_pairs = list(zip(rng.permutation(len(trajs)), rng.permutation(len(trajs))))
    idx_pairs = [pair for pair in idx_pairs if pair[0] != pair[1]]

    pair_data = []
     #TODO: implement pair_trajectories
    for i, j in idx_pairs:
        ret_i = returns[i]
        ret_j = returns[j]

        if temp == 0.0: # Don't use temperature
            if ret_i > ret_j: # prefer i to j
                label = 1.0 
            elif ret_j > ret_i:  # prefer j to i
                label = 0.0
            else:
                label = 0.5 # equal preference
        else:
            prob_i = np.exp(ret_i / temp) / (np.exp(ret_i / temp) + np.exp(ret_j / temp))
            label = prob_i  # Probability that i is preferred

        pair_data.append({'traj1':trajs[i], 'traj2':trajs[j], 'label':label})

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
