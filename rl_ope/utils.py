from data import get_dataset, data_to_sequences, data_to_sequences_rating
from eval_utils import sasrec_model_scoring

from time import time
from functools import reduce
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds, LinearOperator

import numpy as np
import pandas as pd
import torch

import logging
import random

from data import get_dataset, data_to_sequences, SequentialDataset


def process_subseqs(subseqs_h, model_D_sasrec, device):
    states = []
    next_states = []
    scores = []
    actions = []
    ratings = []
    for at, subseq, rating in subseqs_h:
        with torch.no_grad():
            score, state = model_D_sasrec.score_with_state(torch.tensor(subseq, device=device, dtype=torch.long))
            states.append(state.detach().cpu().numpy())

            next_subseq = subseq[1:]
            next_subseq.append(at)
            _, next_state = model_D_sasrec.score_with_state(torch.tensor(next_subseq, device=device, dtype=torch.long))
            next_states.append(next_state.detach().cpu().numpy())
            actions.append(at)
            ratings.append(rating)

    return states, next_states, actions, ratings, scores

def prepare_svd(data, data_description, rank, device):
    """
    Prepares Singular Value Decomposition (SVD) for the interaction matrix.

    Args:
        data (pd.DataFrame): Input data containing user-item interactions.
        data_description (dict): Description of the dataset.
        rank (int): Rank for SVD decomposition.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Item factors matrix from SVD.
    """
    userid = data_description['users']
    itemid = data_description['items']

    n_users = len(data[userid].sort_values().unique())
    n_items = data_description['n_items'] + 2

    interaction_matrix = csr_matrix(
        (
            data['rating'],
            (data[userid], data[itemid]),
        ),
        shape=(n_users, n_items),
        dtype=float
    )

    _, singular_values, vh = svds(
        interaction_matrix,
        k=rank,
        return_singular_vectors='vh'
    )

    sort_order = np.argsort(singular_values)[::-1]
    item_factors = torch.from_numpy(np.ascontiguousarray(vh[sort_order, :].T, dtype=np.float32)).to(device)

    return item_factors

def extract_states_actions(data, model_D_sasrec, n, data_description, device, n_neg_samples=-1, samples_per_user=-1):
    """
    Extracts states, actions, and additional information from the dataset.

    Args:
        data (pd.DataFrame): Input data containing user-item interactions.
        model_D_sasrec (object): Model for generating state representations.
        n (int): Window size for state sequences.
        data_description (dict): Description of the dataset.
        device (torch.device): Device to perform computations on.
        n_neg_samples (int, optional): Number of negative samples per action. Defaults to -1.
        samples_per_user (int, optional): Number of samples per user. Defaults to -1.

    Returns:
        tuple: Extracted states, next states, actions, ratings, sampled sequences, negative actions, and full sequences.
    """
    bs = 1024
    all_actions = np.arange(model_D_sasrec.item_num + 1, dtype=np.int32)
    logger = logging.getLogger("fqe")
    states = []
    next_states = []
    actions = []
    actions_neg = []
    ratings = []

    logger.info("go to data_to_sequences_rating!")
    full_sequences = data_to_sequences_rating(data, data_description, -1)

    states_list = []
    next_states_list = []
    seqs = []
    next_seqs = []
    sampled_seqs = []
    cur_it = 0
    for _, h in tqdm(full_sequences.items(), total=len(full_sequences)):
        seqt = h.copy()
        idxes = random.sample(range(len(seqt)), min(len(seqt), samples_per_user))
        actions += [seqt[idx][0] for idx in idxes]
        ratings += [float(seqt[idx][1]) for idx in idxes]
        seqt = list(list(zip(*seqt))[0])
        state_seqs = [seqt[max(idx - n, 0):idx] for idx in idxes]

        actions_neg.append(np.random.choice(all_actions[~np.isin(all_actions, seqt)],
                                            len(state_seqs) * n_neg_samples,
                                            replace=False).reshape(-1, n_neg_samples))

        state_seqs = [np.pad(seq, (n - len(seq), 0), mode='constant', constant_values=model_D_sasrec.pad_token).tolist() for seq in state_seqs]
        next_state_seqs = [state_seq[1:] + [action] for state_seq, action in zip(state_seqs, actions[-samples_per_user:])]
        seqs += state_seqs
        next_seqs += next_state_seqs
        sampled_seqs += state_seqs
        
        if len(seqs) > bs or cur_it + 1 == len(full_sequences):
            seqs = torch.tensor(seqs, device=device, dtype=torch.long)

            with torch.no_grad():
                states = model_D_sasrec.state_batch(seqs).cpu().detach().numpy()

            states_list.append(states)

            next_seqs = torch.tensor(next_seqs, device=device, dtype=torch.long)
            with torch.no_grad():
                next_states = model_D_sasrec.state_batch(next_seqs).cpu().detach().numpy()

            next_states_list.append(next_states)
            
            seqs = []
            next_seqs = []
            
        cur_it += 1

    return np.concatenate(states_list, axis=0),\
           np.concatenate(next_states_list, axis=0),\
           actions,\
           ratings,\
           sampled_seqs,\
           np.concatenate(actions_neg, axis=0),\
           full_sequences

def process_seq(seqt, model_D_sasrec, device):
    action, seq, rating = seqt

    with torch.no_grad():
        score, state = model_D_sasrec.score_with_state(torch.tensor(seq, device=device, dtype=torch.long))
        state = state.detach().cpu().numpy()
        score = score.detach().cpu().numpy()

    return state, action, rating, score

def extract_states_actions_val(data, model_D_sasrec, n, data_description, device):
    states = []
    actions = []
    scores = []
    ratings = []
    seqs = []

    full_sequences = data_to_sequences_rating(data, data_description)

    for _, seqt in tqdm(full_sequences.items(), total=len(full_sequences)):
        (at, rating) = seqt[-1]
        
        seqt = list(zip(*seqt))[0]
        if len(seqt) < n + 1:
            seqt = np.pad(seqt, (n + 1 - len(seqt), 0), mode='constant', constant_values=model_D_sasrec.pad_token)
            
        seqs.append(seqt[-n-1:-1])

        actions.append(at)
        ratings.append(rating)
        
    states = model_D_sasrec.state_batch(torch.LongTensor(seqs).to(device)).detach().cpu().numpy()
    scores = model_D_sasrec.score_batch(torch.LongTensor(seqs).to(device)).detach().cpu().numpy()

    return states, actions, ratings, scores, np.array(seqs, dtype=np.int32)
