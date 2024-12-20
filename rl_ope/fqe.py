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
    """
    A neural network model for estimating Q-values in reinforcement learning.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        gamma (float): Discount factor for future rewards.
        use_action_emb (bool, optional): Whether to use action embeddings. Defaults to False.
        hidden_size (int, optional): Number of hidden units in each layer. Defaults to 128.
    """
    def __init__(self, state_dim, action_dim, gamma, use_action_emb=False, hidden_size=128):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.use_action_emb = use_action_emb
        self.gamma = gamma
        
        if use_action_emb:
            in_dim = state_dim + action_dim
            out_dim = 1
        else:
            in_dim = state_dim
            out_dim = action_dim

        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_dim)
        self.relu = nn.ReLU()

    def forward(self, state, action=None):
        if self.use_action_emb:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state

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
        return len(self.sampled_seqs)

    def __getitem__(self, idx):
        logger = logging.getLogger("fqe")

        sampled_seq = torch.tensor(self.sampled_seqs[idx], dtype=torch.long)
        action = self.actions[idx]
        next_sampled_seq = torch.zeros(sampled_seq.shape, dtype=torch.long)
        next_sampled_seq[:-1] = sampled_seq[1:]
        next_sampled_seq[-1] = action

        reward = self.rewards[idx]

        actions_neg = torch.from_numpy(self.actions_neg[idx]).long()
        state = torch.from_numpy(self.states[idx]).float()
        next_state = torch.from_numpy(self.next_states[idx]).float()

        return reward, state, next_state, next_sampled_seq, action, actions_neg


class RLDatasetOnlineVal(RLDatasetOnline):
    def __init__(self, config):
        super().__init__(config)

        self.seqs = config["seqs"]
        self.states = config["states"]

    def __len__(self):
        return len(self.full_sequences)

    def get_seq(self, idx):
        state = torch.from_numpy(self.seqs[idx])
    
        return state


def collate_fn(batch):
    batch_rewards,\
    batch_states,\
    batch_next_states,\
    batch_next_sampled_seqs,\
    batch_actions,\
    batch_actions_neg = list(zip(*batch))
    return torch.tensor(batch_rewards).float(),\
           torch.stack(batch_states, dim=0),\
           torch.stack(batch_next_states, dim=0),\
           torch.stack(batch_next_sampled_seqs, dim=0),\
           torch.tensor(batch_actions).long(),\
           torch.stack(batch_actions_neg, dim=0)
    

class FQE:
    """
    Fitted Q Evaluation (FQE) class for reinforcement learning policy evaluation.

    Args:
        dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        pi_e (object): Policy being evaluated.
        item_emb (torch.Tensor): Embedding matrix for items.
        optim_conf (dict): Optimizer configuration.
        n_epochs (int): Number of training epochs.
        state_dim (int): Dimension of state representation.
        n_actions (int): Total number of actions.
        action_dim (int): Dimension of action embeddings.
        hidden_size (int): Number of hidden units in each layer.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update rate for target network.
        n_sampled_actions (int): Number of sampled actions during training.
        use_action_emb (bool): Whether to use action embeddings.
        device (torch.device): Device to use for computations.
    """
    def __init__(self,
                 dataset,
                 val_dataset,
                 pi_e,
                 item_emb,
                 optim_conf,
                 n_epochs,
                 state_dim,
                 n_actions,
                 action_dim,
                 hidden_size,
                 gamma,
                 tau,
                 n_sampled_actions,
                 use_action_emb,
                 device):
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.pi_e = pi_e
        self.optim_conf = optim_conf
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.tau = tau
        self.device = device
        self.n_sampled_actions = n_sampled_actions
        self.use_action_emb = use_action_emb
        
        self.item_emb = item_emb

        self.q = QNetwork(self.state_dim,
                          self.action_dim,
                          self.gamma,
                          self.use_action_emb,
                          self.hidden_size).to(self.device)

    @staticmethod
    def copy_over_to(source, target):
        """
        Copies parameters from the source network to the target network.
        """
        target.load_state_dict(source.state_dict())

    @staticmethod
    def soft_update(source, target, tau):
        """
        Performs a soft update of the target network parameters.
        """
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    tau * source_param.data + (1.0 - tau) * target_param.data
                )

    def train(self, batch_size, plot_info=True):
        """
        Trains the Q-network using the provided dataset.
        """
        logger = logging.getLogger("fqe")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.q.parameters()), **self.optim_conf)
        q_prev = QNetwork(self.state_dim,
                          self.action_dim,
                          self.gamma,
                          self.use_action_emb,
                          self.hidden_size).to(self.device)

        self.copy_over_to(self.q, q_prev)

        val_idxes = np.random.choice(np.arange(len(self.val_dataset)), 200)
        val_states_seq = [self.val_dataset.get_seq(idx) for idx in val_idxes]
        val_states = [torch.from_numpy(self.val_dataset.states[idx]).float() for idx in val_idxes]
        
        wandb.watch(self.q, log_freq=100)

        values = []
        for epoch in range(self.n_epochs):
            dataloader = DataLoader(self.dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=collate_fn,
                                    num_workers=16,
                                    prefetch_factor=10,
                                    drop_last=True)

            loss_history = []
            cur_it = 0
            for rewards, states, next_states, next_state_seqs, actions, actions_neg in tqdm(dataloader, total=len(dataloader)):
                rewards = rewards.to(self.device)
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                next_state_seqs = next_state_seqs.to(self.device)
                actions = actions.to(self.device)
                actions_neg = actions_neg.to(self.device)

                with torch.no_grad():
                    logits = self.pi_e.score_batch(next_state_seqs)
                    pi_e_s = torch.softmax(logits, 1)
                    
                    if self.use_action_emb:
                        sampled_actions = torch.multinomial(pi_e_s, self.n_sampled_actions, replacement=True) #(bs, n_sampled_actions)           
                        emb_actions = self.item_emb[sampled_actions] #(bs, n_sampled_actions, action_dim)
                        next_states = next_states.unsqueeze(1).expand(-1, self.n_sampled_actions, self.action_dim)

                        q_vals = q_prev(next_states, emb_actions).mean(1).squeeze(1) #(bs)
                    else:
                        q_vals = q_prev(next_states)
                        q_vals = (pi_e_s * q_vals).sum(axis=-1)

                    y = rewards + self.gamma * q_vals
                    y_neg = self.gamma * q_vals # (batch_size)

                preds = self.predict(states, actions.unsqueeze(-1)).squeeze(-1) # (bs)
                preds_neg = self.predict(states, actions_neg) # (bs, n_neg_samples)

                assert len(preds.shape) == 1
                assert len(y.shape) == 1

                loss = torch.sum((preds - y)**2) + torch.sum((preds_neg - y_neg.unsqueeze(-1))**2)
                loss = loss / (preds.numel() + preds_neg.numel())
                optimizer.zero_grad()
                loss.backward()
                
                wandb.log({"loss": loss.item()})

                torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
                optimizer.step()

                loss_history.append(loss.item())
                if self.tau > 0:
                    self.soft_update(self.q, q_prev, self.tau)
                
                cur_it += 1

            val_probs = [torch.softmax(self.pi_e.score_with_state(state_seq.to(self.device))[0].unsqueeze(0), 1).squeeze(0).detach().cpu() for state_seq in val_states_seq]
            val_actions = [np.random.choice(np.arange(self.n_actions), p=prob.numpy()) for prob in val_probs]
            val_preds = self.predict(torch.stack(val_states, dim=0).to(self.device),
                                     torch.tensor(val_actions, device=self.device, dtype=torch.long).unsqueeze(-1)).detach()

            values.append(torch.mean(val_preds).item())

            if plot_info:
                self.plot_info(loss_history, values)

            logger.info(f"Last iter loss = {loss_history[-1]}, value on val = {values[-1]}")
            
            wandb.log({"value on val": values[-1]})

            if self.tau > 0:
                self.soft_update(self.q, q_prev, self.tau)
            else:
                self.copy_over_to(self.q, q_prev)

            logger.info(f"Finished Epoch {epoch}.")

        return values

    def predict(self, states, actions):
        """
        Predicts Q-values for the given states and actions.
        """
        if self.use_action_emb:
            n_actions = actions.shape[1]
            emb_actions = self.item_emb[actions].detach()
            q_vals = self.q(states.unsqueeze(1).expand(-1, n_actions, self.action_dim), emb_actions).squeeze(-1) #(bs, n_actions)
        else:
            q_vals = torch.take_along_dim(self.q(states), actions, dim=1)

        return q_vals

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