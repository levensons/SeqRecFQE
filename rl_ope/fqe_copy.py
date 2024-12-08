from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
import wandb

import logging

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, gamma, hidden_size=128):
        super().__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.gamma = gamma

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        # q_vals = self.model(x)
        q_vals = self.relu(self.fc1(x))
        q_vals = self.relu(self.fc2(q_vals))
        q_vals = self.fc3(q_vals)

        return q_vals


class RLDatasetOnline(Dataset):
    def __init__(self, config):
        self.states = config["states"]

        if "next_states" in config:
            self.next_states = config["next_states"]

        self.actions = config["actions"]
        self.rewards = config["rewards"]
        self.full_sequences = list(config["full_sequences"].items())
        self.n = config["n"]
        self.pad_token = config["pad_token"]
        
        if "samples_per_user" in config:
            self.samples_per_user = config["samples_per_user"]

        if "sampled_seqs" in config:
            self.sampled_seqs = config["sampled_seqs"]
            
        if "actions_neg" in config:
            self.actions_neg = config["actions_neg"]

        if "n_neg_samples" in config:
            self.n_neg_samples = config["n_neg_samples"]

        if "all_actions" in config:
            self.all_actions = config["all_actions"]

    def __len__(self):
        return len(self.full_sequences)

    def __getitem__(self, idx):
        logger = logging.getLogger("fqe")

        # start_time = time.time()

        state_seq = list(zip(*self.full_sequences[idx][1]))[0]
        sampled_seqs = torch.tensor(self.sampled_seqs[idx], dtype=torch.long)
        actions = torch.tensor(self.actions[idx], dtype=torch.long)
        # state_seq, action = self.get_state_action(idx)
        next_sampled_seqs = torch.zeros(sampled_seqs.shape, dtype=torch.long)
        next_sampled_seqs[:, :-1] = sampled_seqs[:, 1:]
        next_sampled_seqs[:, -1] = actions
        
        # end_time = time.time()  # Record the end time
        # elapsed_time = end_time - start_time
        # logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")

        # next_state_seq = nn.functional.pad(next_state_seq, (self.n - len(next_state_seq), 0), mode='constant', value=self.pad_token)

        rewards = torch.tensor(self.rewards[idx]).float()

        # actions_neg = torch.tensor(np.random.choice(self.all_actions[~np.isin(self.all_actions, state_seq)],
        #                                             self.samples_per_user * self.n_neg_samples,
        #                                             replace=False), dtype=torch.long).reshape(-1, self.n_neg_samples)

        # actions_neg = torch.tensor(random.sample(range(len(self.all_actions)),
        #                                          self.samples_per_user * self.n_neg_samples),
        #                            dtype=torch.long).reshape(-1, self.n_neg_samples)

        idx_start = idx * self.samples_per_user
        idx_end = (idx + 1) * self.samples_per_user
        actions_neg = torch.from_numpy(self.actions_neg[idx_start:idx_end]).long()
        states = torch.from_numpy(self.states[idx_start:idx_end]).float()
        next_states = torch.from_numpy(self.next_states[idx_start:idx_end]).float()

        return rewards, states, next_states, next_sampled_seqs, actions, actions_neg

class RLDatasetOnlineVal(RLDatasetOnline):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.full_sequences)

    def get_seq(self, idx):
        if self.n < 0:
            seq = torch.tensor(self.full_sequences[idx][1][:-1], dtype=torch.long)
        else:
            seq = self.full_sequences[idx][1]
            if len(seq) < self.n + 1:
                seq = np.pad(seq, (self.n - len(seq), 0), mode='constant', constant_values=self.pad_token)

            seq = torch.tensor(seq[-self.n-1:-1], dtype=torch.long)

        return seq


def collate_fn(batch):
    # batch_rewards = []
    # batch_states = []
    # batch_next_states = []
    # next_samples_seqs = []
    # actions = []
    # actions_neg = []
    
    batch_rewards,\
    batch_states,\
    batch_next_states,\
    batch_next_sampled_seqs,\
    batch_actions,\
    batch_actions_neg = list(zip(*batch))
    # for rewards, states, next_states, next_sampled_seqs, actions, actions_neg in batch:
    #     batch_rewards.append(rewards)
    #     batch_states.append(states)
    #     batch_next_states.append(next_states)
    #     batch_next_sampled_seqs.append(next_sampled_seqs)
    #     batch_actions.append(actions)
    #     batch_actions_neg.append(actions_neg)

    return torch.cat(batch_rewards, dim=0),\
           torch.cat(batch_states, dim=0),\
           torch.cat(batch_next_states, dim=0),\
           torch.cat(batch_next_sampled_seqs, dim=0),\
           torch.cat(batch_actions, dim=0),\
           torch.cat(batch_actions_neg, dim=0)
    

class FQE:
    def __init__(self,
                 dataset,
                 val_dataset,
                 pi_e,
                 optim_conf,
                 n_epochs,
                 state_dim,
                 n_actions,
                 hidden_size,
                 gamma,
                 tau,
                 device):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.pi_e = pi_e
        self.optim_conf = optim_conf
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.tau = tau
        self.device = device

        self.pi_e = self.pi_e.to(self.device)
        self.pi_e.eval()

        self.q = QNetwork(self.state_dim,
                          self.n_actions,
                          self.gamma,
                          self.hidden_size).to(self.device)
        
        # for param in self.q.fc3.parameters():
        #     param.requires_grad = False
        
        # self.q = nn.DataParallel(self.q)

    @staticmethod
    def copy_over_to(source, target):
        target.load_state_dict(source.state_dict())

    @staticmethod
    def soft_update(source, target, tau):
        """
        Perform a soft update of the target network parameters.

        Args:
            target_network (torch.nn.Module): The target network.
            current_network (torch.nn.Module): The current network.
            tau (float): The soft update rate (0 < tau <= 1).
        """
        with torch.no_grad():  # No gradient computation for this operation
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    tau * source_param.data + (1.0 - tau) * target_param.data
                )

    def train(self, batch_size, plot_info=True):
        logger = logging.getLogger("fqe")
        # optimizer = optim.Adam(self.q.parameters(), **self.optim_conf)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.q.parameters()), **self.optim_conf)
        q_prev = QNetwork(self.state_dim,
                          self.n_actions,
                          self.gamma,
                          self.hidden_size).to(self.device)

        self.copy_over_to(self.q, q_prev)

        val_idxes = np.random.choice(np.arange(len(self.val_dataset)), 200)
        val_states_seq = [self.val_dataset.get_seq(idx) for idx in val_idxes]
        val_states = [torch.from_numpy(self.val_dataset.states[idx]).float() for idx in val_idxes]
        
        wandb.watch(self.q, log_freq=100)

        values = []
        for epoch in range(self.n_epochs):
            dataloader = DataLoader(self.dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=16, drop_last=True)
            # dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

            loss_history = []
            cur_it = 0
            for rewards, states, next_states, next_state_seqs, actions, actions_neg in tqdm(dataloader, total=len(dataloader)):
                rewards = rewards.to(self.device)
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                next_state_seqs = next_state_seqs.to(self.device)
                actions = actions.to(self.device)
                actions_neg = actions_neg.to(self.device)

                next_state_seqs = next_state_seqs[:, -self.pi_e.pos_emb.num_embeddings:]

                with torch.no_grad():
                    logits = self.pi_e.score_batch(next_state_seqs)
                    pi_e_s = torch.softmax(logits, 1)

                    q_vals = q_prev(next_states)

                    q_vals = (pi_e_s * q_vals).sum(axis=-1)

                    y = rewards + self.gamma * q_vals
                    y_neg = self.gamma * q_vals # (batch_size)

                preds = self.predict(states, actions.unsqueeze(-1)).squeeze(-1) # (batch_size)
                preds_neg = self.predict(states, actions_neg) # (batch_size, n_neg_samples)

                assert len(preds.shape) == 1
                assert len(y.shape) == 1

                loss = torch.sum((preds - y)**2) + torch.sum((preds_neg - y_neg.unsqueeze(-1))**2)
                loss = loss / (preds.numel() + preds_neg.numel())
                optimizer.zero_grad()
                loss.backward()
                
                wandb.log({"loss": loss.item()})

                torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
                optimizer.step()

                # loss_history.append(loss.item())
                if (cur_it + 1) % 5 == 0:
                    self.soft_update(self.q, q_prev, self.tau)
                    loss_history.append(loss.item())
                    
                # if loss.item() > 40:
                #     logger.info(actions)
                #     logger.info(actions_neg)
                
                cur_it += 1

            val_probs = [torch.softmax(self.pi_e.score(state_seq.to(self.device)), 1).squeeze(0).detach().cpu() for state_seq in val_states_seq]
            val_actions = [np.random.choice(np.arange(self.n_actions), p=prob.numpy()) for prob in val_probs]
            val_preds = self.predict(torch.stack(val_states, dim=0).to(self.device),
                                     torch.tensor(val_actions, device=self.device, dtype=torch.long).unsqueeze(-1)).detach()

            values.append(torch.mean(val_preds).item())

            if plot_info:
                self.plot_info(loss_history, values)

            logger.info(f"Last iter loss = {loss_history[-1]}, value on val = {values[-1]}")
            
            wandb.log({"value on val": values[-1]})

            # self.copy_over_to(self.q, q_prev)
            # self.soft_update(self.q, q_prev, self.tau)

            logger.info(f"Finished Epoch {epoch}.")


        return values

    def predict(self, states, actions):
        return torch.take_along_dim(self.q(states), actions, dim=1)

    def plot_info(self, loss_history, values):
        fig = plt.figure(figsize=(20, 10))

        fig.add_subplot(1, 2, 1)
        plt.plot(loss_history[::10])
        plt.yscale("log")
        plt.grid(True)

        fig.add_subplot(1, 2, 2)
        plt.plot(values)
        plt.grid(True)

        plt.savefig("plot.png")
        plt.show()














