import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import validate_model
from model import ContinuousPolicy
import gymnasium as gym
from tqdm import tqdm
from batch_collection import collect_pair_data
import argparse
import yaml
class PairedTrajectoryDataset(Dataset):
    def __init__(self, pair_data):
        self.data = pair_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = self.data[idx]
        t1, t2 = p["traj1"], p["traj2"]
        return {
            "traj1_state": torch.from_numpy(t1[0]).float(),  # (T, obs_dim)
            "traj1_act": torch.from_numpy(t1[2]).float(),  # (T, act_dim)
            "traj1_logp": torch.from_numpy(t1[3]).float(),
            "traj2_state": torch.from_numpy(t2[0]).float(),
            "traj2_act": torch.from_numpy(t2[2]).float(),
            "traj2_logp": torch.from_numpy(t2[3]).float(),
            "label": torch.tensor(p["label"], dtype=torch.float32),
        }

    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}


class DPOTrainer:
    def __init__(self, env, policy, optimizer, beta=1, batch_size=16, device="cpu"):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.beta = beta
        self.batch_size = batch_size
        self.device = device

    def _evaluate(self, seed, n_trajs=40):
        mean_rew, std_rew = validate_model(self.policy, self.env, n_trajs)
        return mean_rew, std_rew

    def train(self, pair_data, num_epochs_per_iter=6, num_iterations=10, seed=None):
        # TODO: implement DPO training
        n = len(pair_data)
        curr = 0


        for iteration in range(num_iterations):
            policy_ref = self.policy
            # if not first iteration, do iterative DPO
            # otherwise, just do normal DPO
            for epoch in range(num_epochs_per_iter):
                if curr == n:
                    break
                traj_0, traj_1, label = pair_data[curr]

                print(len(traj_0[0])) # 3
                
                logprob_0 = 0
                state, reward, action = traj_0
                logprob_0 += self.beta * (self.policy.compute_log_likelihood(torch.Tensor(state), torch.Tensor(action)))
                logprob_0 -= self.beta* (policy_ref.compute_log_likelihood(torch.Tensor(state), torch.Tensor(action)))

                logprob_1 = 0
                state, reward, action = traj_1
                logprob_1 += self.beta * (self.policy.compute_log_likelihood(torch.Tensor(state), torch.Tensor(action)))
                logprob_1 -= self.beta* (policy_ref.compute_log_likelihood(torch.Tensor(state), torch.Tensor(action)))
                loss = logprob_0 + logprob_1
                loss -= label
                loss = loss**2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                curr += 1
            
                
                
            


def main():
    env = gym.make("Swimmer-v5")
    policy = ContinuousPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    # load model
    policy.load_state_dict(torch.load("swimmer_checkpoint.pt", weights_only=False))
    pair_data = torch.load("pair_data.pt", weights_only=False)

    # argparse
    # load hparams
    with open("hparam.yaml", "r") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(hparams["lr"]))
    dpo = DPOTrainer(env, policy, optimizer, float(hparams["beta"]), int(hparams["batch_size"]))
    
    if hparams["iterative_dpo"]:
        iterations = 10
    else:
        iterations = 1

    dpo.train(pair_data, num_iterations=iterations, seed=42, num_epochs_per_iter=hparams["num_epochs_per_iter"])
    mean_rew, std_rew = validate_model(policy)
    print(f"Mean reward of trajectories: {mean_rew}, std: {std_rew}")
    torch.save(policy, "dpo.pt")

if __name__ == "__main__":
    main()
