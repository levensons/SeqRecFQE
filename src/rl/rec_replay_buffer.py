from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import torch

TensorBatch = List[torch.Tensor]

class RecReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        sampler = None
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._device = device
        self._sampler = sampler

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def sample(self, batch_size: int) -> TensorBatch:
        for _, *seq_data in self._sampler:
            break
        seq, pos, neg = (torch.tensor(np.array(x), device=self._device, dtype=torch.long) for x in seq_data)
        rewards = torch.ones_like(pos).to(self._device)
        dones = torch.zeros_like(pos).to(self._device)
        dones[:,-1] = 1.0
        next_states = pos

        return [seq, pos, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError