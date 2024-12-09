from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch.nn import functional as F
import pickle

DEVICE = torch.device("cuda")

def batch_to_device(batch):
    new_batch = {key: value.to(DEVICE) for key, value in batch.items()}
    return new_batch

def load_model(path_to_model):
    return torch.load(path_to_model)

def make_rsa(item_seq, memory_size, item_num, inference=False):
    if inference:
        return {
            "rtgs": torch.arange(len(item_seq) + 1, 0, -1)[..., None],
            "states": F.pad(item_seq, (memory_size, 0), value=item_num).unfold(0, 3, 1),
            "actions": item_seq[..., None],
            "timesteps": torch.tensor([[0]]),
            "users": torch.tensor([0]),
        }
    return {
        "rtgs": torch.arange(len(item_seq), 0, -1)[..., None],
        "states": F.pad(item_seq, (memory_size, 0), value=item_num).unfold(0, 3, 1)[
            :-1
        ],
        "actions": item_seq[..., None],
        "timesteps": torch.tensor([[0]]),
        "users": torch.tensor([0]),
    }


class SeqsDataset(Dataset):
    def __init__(self, seqs, item_num):
        self.seqs = seqs
        self.item_num = item_num

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if len(seq) == 0:
            seq = torch.tensor([self.item_num, self.item_num, self.item_num])
        return make_rsa(self.seqs[idx], 3, self.item_num, True)

    def __len__(self):
        return len(self.seqs)


def seq_to_states(model, seqs):
    item_num = model.config.vocab_size
    seqs_dataset = SeqsDataset(seqs, item_num)
    seqs_dataloader = DataLoader(seqs_dataset, batch_size=128, num_workers=4)

    outputs = []
    # for batch in tqdm(seqs_dataloader, total=len(seqs_dataloader)):
    for batch in seqs_dataloader:
        batch = batch_to_device(batch)
        trajectory_len = batch["states"].shape[1]
        state_embeddings = model.state_repr(
            batch["users"].repeat((1, trajectory_len)).reshape(-1, 1),
            batch["states"].reshape(-1, 3),
        )

        state_embeddings = state_embeddings.reshape(
            batch["states"].shape[0], batch["states"].shape[1], model.config.n_embd
        )
        outputs.append(state_embeddings[:, -1])

    return torch.cat(outputs, dim=0)


def seq_to_logits(model, seqs):
    item_num = model.config.vocab_size
    seqs_dataset = SeqsDataset(seqs, item_num)
    seqs_dataloader = DataLoader(seqs_dataset, batch_size=128, num_workers=4)

    outputs = []
    # for batch in tqdm(seqs_dataloader, total=len(seqs_dataloader)):
    for batch in seqs_dataloader:
        batch = batch_to_device(batch)
        outputs.append(model(**batch).detach()[:, -1])

    return torch.cat(outputs, dim=0)
