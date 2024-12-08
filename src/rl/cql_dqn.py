#https://github.com/BY571/CQL/blob/main/CQL-DQN/agent.py
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_alpha: float = 10.0  # Minimal Q weight
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

    # AntMaze hacks
    bc_steps: int = int(0)  # Number of BC steps at start
    reward_scale: float = 5.0
    reward_bias: float = -1.0
    policy_log_std_multiplier: float = 1.0

    # Wandb logging
    project: str = "CORL"
    group: str = "CQL-D4RL"
    name: str = "CQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init
        self.hidden_dim = hidden_dim

        layers = [
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

class DQNCQL:
    def __init__(
        self,
        body,
        body_optimizer,
        q_1,
        q_1_optimizer,
        q_2,
        q_2_optimizer,
        target_entropy: float,
        discount: float = 0.99,
        alpha_multiplier: float = 1.0,
        use_automatic_entropy_tuning: bool = True,
        backup_entropy: bool = False,
        policy_lr: bool = 3e-4,
        qf_lr: bool = 3e-4,
        soft_target_update_rate: float = 5e-3,
        bc_steps=100000,
        target_update_period: int = 1,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        cql_negative_samples: int = 10,
        device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_negative_samples = cql_negative_samples
        self._device = device

        self.total_it = 0

        self.body = body
        self.q_1 = q_1
        self.q_2 = q_2

        self.target_q_1 = deepcopy(self.q_1).to(device)
        self.target_q_2 = deepcopy(self.q_2).to(device)

        self.body_optimizer = body_optimizer
        self.q_1_optimizer = q_1_optimizer
        self.q_2_optimizer = q_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_q_1, self.q_1, soft_target_update_rate)
        soft_update(self.target_q_2, self.q_2, soft_target_update_rate)

    def sample_negative_actions_with_prob(self, output, real_actions):
        K = self.cql_negative_samples
        batch_size, num_actions = output.shape
        probs = F.softmax(output, dim=1)

        # Zero out the probabilities of the real actions
        mask = torch.arange(num_actions).expand(batch_size, num_actions).to(self._device) == real_actions.unsqueeze(1)
        probs = probs.masked_fill(mask, 0)
        
        # Normalize the probabilities again
        probs = F.softmax(probs, dim=1)

        negative_samples = []
        for i in range(batch_size):
            sampled_actions = torch.multinomial(probs[i], K, replacement=False)
            negative_samples.append(sampled_actions)

        return torch.stack(negative_samples)
    
    def sample_actions_uniform(self, real_actions):
#         M = self.cql_n_actions+2
#         K = self.cql_negative_samples
#         N = real_actions.shape[0]
#         all_actions = torch.arange(1, M+1).to(self._device)

#         samples = torch.empty(N, K, dtype=torch.long).to(self._device)
#         # samples = torch.randint(1, M+1, size=(N, K)).to(self._device)
#         for i in range(N):
#             possible_actions = all_actions[all_actions != real_actions[i]]
#             sampled_indices = torch.randperm(possible_actions.size(0))[:K].to(self._device)
#             samples[i] = possible_actions[sampled_indices]

#         return samples

        M = self.cql_n_actions + 2  # Total number of actions
        K = self.cql_negative_samples  # Number of negative samples to draw
        N = real_actions.shape[0]  # Batch size

        # Create a matrix of all possible actions, excluding real_actions
        all_actions = torch.arange(1, M + 1, device=self._device).unsqueeze(0).repeat(N, 1)  # Shape: [N, M]
        mask = all_actions != real_actions.unsqueeze(1)  # Mask real actions (Shape: [N, M])

        # Mask out the real actions and select only possible actions
        possible_actions = torch.masked_select(all_actions, mask).view(N, -1)  # Shape: [N, M-1]

        # Randomly sample K actions for each batch
        sampled_indices = torch.randint(0, possible_actions.size(1), (N, K), device=self._device)  # Shape: [N, K]
        samples = torch.gather(possible_actions, 1, sampled_indices)  # Gather sampled actions
        
        return samples

    def score_with_state(self, seq):
        return self.body.score_with_state(seq)

    def score(self, seq):
        body_out = self.score_with_state(seq)[-1]
        body_out = body_out.reshape(-1, body_out.shape[-1])
        predictions = (self.q_1(body_out) + self.q_2(body_out)) / 2.0
        return predictions

    def score_batch(self, log_seqs):
        emb = self.log2feats(log_seqs)
        body_out = emb[:, -1, :]

        predictions = (self.q_1(body_out) + self.q_2(body_out)) / 2.0
        return predictions
    
    def state_batch(self, log_seqs):
        return self.log2feats(log_seqs)[:, -1]

    def log2feats(self, log_seqs):
        return self.body.log2feats(log_seqs)

    def _bc_loss(
        self,
        q_pred,
        actions
    ):
        q_softmax = torch.softmax(q_pred, dim=1)
        actions = actions.reshape(-1).unsqueeze(1)
        pos_prob = q_softmax.gather(1, actions).squeeze(1)

        neg_actions = self.sample_negative_actions(
            q_pred.detach(),
            actions
        )
        q1_negatives = []
        for i in range(self.cql_negative_samples):
            q1_negatives.append(
                q_softmax.gather(1, neg_actions[:,i]).unsqueeze(1)
            )
        neg_prob = torch.cat(q1_negatives, dim=1).reshape(-1)
        gt = torch.cat([
            torch.ones_like(pos_prob),
            torch.zeros_like(neg_prob)
        ])

        return torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat([pos_prob, neg_prob]),
            gt
        )
    

    def _cql_loss(self, q_values, current_action, q_negatives):
        """Computes the CQL loss for a batch of Q-values and actions."""
        q_cat = torch.cat([q_values, q_negatives], dim=1)
        # print(q_cat.shape)
        # print(self.cql_temp.shape)
        # assert False
        logsumexp = torch.logsumexp(q_cat / self.cql_temp, dim=1, keepdim=True) * self.cql_temp
        q_a = q_values.gather(1, current_action)
        return (logsumexp - q_a).mean()
    
#     def _cql_loss(self, q_values, current_action, q_negatives):
#         chunk_size = 1024  # Adjust based on GPU memory
#         num_chunks = q_values.size(1) // chunk_size + (q_values.size(1) % chunk_size > 0)

#         # Stabilize with max value
#         max_vals, _ = torch.max(q_values, dim=1, keepdim=True)

#         logsumexp_chunks = []
#         for i in range(num_chunks):
#             start = i * chunk_size
#             end = min(start + chunk_size, q_values.size(1))
#             chunk = q_values[:, start:end] - max_vals
#             logsumexp_chunks.append(torch.sum(torch.exp(chunk), dim=1, keepdim=True))

#         # Combine results from chunks
#         logsumexp = max_vals + torch.log(torch.cat(logsumexp_chunks, dim=1).sum(dim=1, keepdim=True))
#         q_a = q_values.gather(1, current_action)
#         return (logsumexp - q_a).mean()

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        body_out = self.body(observations, None, None)

        with torch.no_grad():
            next_body_out = self.body(next_observations, None, None)
            target_out = torch.min(self.target_q_1(next_body_out), self.target_q_2(next_body_out))
            Q_target_next = target_out.\
                detach().\
                max(1)[0].\
                unsqueeze(1)

            Q_targets = rewards[:, -1].unsqueeze(1) + (self.discount * Q_target_next * (1-dones[:, -1].unsqueeze(1)))

        # actions = actions.reshape(-1).unsqueeze(1)
        actions = actions[:, -1].unsqueeze(1)
        q1_predicted = self.q_1(body_out)
        q2_predicted = self.q_2(body_out)
        q1_expected = q1_predicted.gather(1, actions)
        q2_expected = q2_predicted.gather(1, actions)

        q1_td_loss = F.mse_loss(q1_expected, Q_targets)
        q2_td_loss = F.mse_loss(q2_expected, Q_targets)

        # CQL
        with torch.no_grad():
            """neg_actions = self.sample_negative_actions(
                torch.min(q1_predicted,q2_predicted).detach(),
                actions
            )"""
            neg_actions = self.sample_actions_uniform(actions)

        # q1_negatives = []
        # q2_negatives = []
        # for i in range(self.cql_negative_samples):
        #     q1_negatives.append(
        #         q1_predicted.gather(1, neg_actions[:,i])
        #     )
        #     q2_negatives.append(
        #         q2_predicted.gather(1, neg_actions[:,i])
        #     )

        q1_negatives = q1_predicted.gather(1, neg_actions)
        q2_negatives = q2_predicted.gather(1, neg_actions)
        # q1_negatives = torch.cat(q1_negatives, dim=1)
        # q2_negatives = torch.cat(q2_negatives, dim=1)

        q1_cql_loss = self._cql_loss(q1_predicted, actions, q1_negatives)
        q2_cql_loss = self._cql_loss(q2_predicted, actions, q2_negatives)

        qf_loss = q1_td_loss + q2_td_loss + self.cql_alpha * (q1_cql_loss + q2_cql_loss)

        return qf_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        self.total_it += 1

        """ Q function loss """
        log_dict = dict()
        loss = torch.tensor(0.0).to(self._device)
        
        if self.total_it > self.bc_steps or True:
            qf_loss = self._q_loss(
                observations, actions, next_observations, rewards, dones, 0, log_dict
            )
            loss += qf_loss

        #if self.use_automatic_entropy_tuning:
        #    self.alpha_optimizer.zero_grad()
        #    self.alpha_optimizer.step()

        log_dict.update(
            loss = loss.item()
        )

        self.body_optimizer.zero_grad()
        self.q_1_optimizer.zero_grad()
        self.q_2_optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        self.q_1_optimizer.step()
        self.q_2_optimizer.step()
        self.body_optimizer.step()
        

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "body": self.body.state_dict(),
            "q_1": self.q_1.state_dict(),
            "q_2": self.q_2.state_dict(),
            "q1_target": self.target_q_1.state_dict(),
            "q2_target": self.target_q_2.state_dict(),
            "critic_1_optimizer": self.q_1_optimizer.state_dict(),
            "critic_2_optimizer": self.q_1_optimizer.state_dict(),
            "body_optim": self.body_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.body.load_state_dict(state_dict=state_dict["body"])
        self.q_1.load_state_dict(state_dict=state_dict["q_1"])
        self.q_2.load_state_dict(state_dict=state_dict["q_2"])

        self.target_q_1.load_state_dict(state_dict=state_dict["q1_target"])
        self.target_q_2.load_state_dict(state_dict=state_dict["q2_target"])

        self.q_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.q_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.body_optimizer.load_state_dict(state_dict=state_dict["body_optim"])

        self.total_it = state_dict["total_it"]