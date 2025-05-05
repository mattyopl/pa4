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
        pairedTrajData = PairedTrajectoryDataset(pair_data)
        dataloader = DataLoader(pairedTrajData, batch_size=self.batch_size, shuffle=True, collate_fn=PairedTrajectoryDataset.collate_fn)

        for iteration in range(num_iterations):
            policy_ref = self.policy
            # if not first iteration, do iterative DPO
            # otherwise, just do normal DPO
            for epoch in tqdm(range(num_epochs_per_iter), desc="Running epochs", leave=True):
                    for batch in tqdm(dataloader, desc="Running epochs", leave=True):
                        traj1_state = batch["traj1_state"].to(self.device)   # (B, T, obs_dim)
                        traj1_act   = batch["traj1_act"].to(self.device)     # (B, T, act_dim)
                        traj1_logp  = batch["traj1_logp"].to(self.device)    # (B, T)
                        traj2_state = batch["traj2_state"].to(self.device)
                        traj2_act   = batch["traj2_act"].to(self.device)
                        traj2_logp  = batch["traj2_logp"].to(self.device)
                        label       = batch["label"].to(self.device)         # (B,)
                        total_loss = 0

                        for i in range(self.batch_size):
                            logprob_1 = 0
                            logprob_1 += self.beta * (torch.sum(traj1_logp[i]))
                            for j in range(len(traj1_state[i])):
                                logprob_1 -= self.beta * (policy_ref.compute_log_likelihood(traj1_state[i][j], traj1_act[i][j]))

                            logprob_2 = 0
                            logprob_2 += self.beta * (torch.sum(traj2_logp[i]))
                            for j in range(len(traj1_state[i])):
                                logprob_2 -= self.beta * (policy_ref.compute_log_likelihood(traj2_state[i][j], traj2_act[i][j]))
                            loss = logprob_1 + logprob_2
                            loss -= label[i]
                            loss = loss**2
                            total_loss += loss

                        total_loss = total_loss/self.batch_size
                        print(total_loss.shape)
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        self.optimizer.step()
            
                
                
            


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
    mean_rew, std_rew = validate_model(policy, env)
    print(f"Mean reward of trajectories: {mean_rew}, std: {std_rew}")
    torch.save(policy, "dpo.pt")

if __name__ == "__main__":
    main()
