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

from data import get_dataset, data_to_sequences
from concurrent.futures import ProcessPoolExecutor
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


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
    userid = data_description['users']
    itemid = data_description['items']

    n_users = len(data[userid].sort_values().unique())
    n_items = data_description['n_items'] + 2 # +1 for the pad item of SasRec and +1 for &@!&%$

    interaction_matrix = csr_matrix(
                                        (data['rating'],
                                        (data[userid], data[itemid]), #subtract 1 to start with [0,1,2,...]
                                    ),
                                    shape=(n_users, n_items),
                                    dtype=float)

    _, singular_values, vh = svds(
        interaction_matrix,
        k=rank,
        return_singular_vectors='vh'
    )

    sort_order = np.argsort(singular_values)[::-1]
    item_factors = torch.from_numpy(np.ascontiguousarray(vh[sort_order, :].T, dtype=np.float32)).to(device)

    return item_factors

def process_seqs(model, seqs, next_seqs):
    seqs = torch.tensor(seqs, dtype=torch.long)[:, -model.num_embeddings:]

    with torch.no_grad():
        states = model.state_batch(seqs).cpu().detach().numpy().astype(np.float32)

    next_seqs = torch.tensor(next_seqs, dtype=torch.long)[:, -model.num_embeddings:]
    with torch.no_grad():
        next_states = model.state_batch(next_seqs).cpu().detach().numpy().astype(np.float32)
        next_scores = model.score_batch(next_seqs).cpu().detach().numpy().astype(np.float16)
    
    return states, next_states, next_scores
    

def extract_states_actions(data, model_D_sasrec, n, data_description, device, n_neg_samples=-1, samples_per_user=-1):
    bs = 1024
    all_actions = np.arange(model_D_sasrec.item_num + 1, dtype=np.int32)
    logger = logging.getLogger("fqe")
    model_D_sasrec.item_emb = model_D_sasrec.item_emb.to(device)
    states = []
    next_states = []
    actions = []
    actions_neg = []
    ratings = []

    logger.info("go to data_to_sequences_rating!")
    full_sequences = data_to_sequences_rating(data, data_description, -1)
    # logger.info(full_sequences.head())
    # full_sequences.to_csv('./mdp/full_sequences.csv')

    # full_sequences = pd.read_csv("./mdp/full_sequences.csv", index_col=0, squeeze=True).rename(None).apply(eval)

    states_list = []
    next_states_list = []
    next_scores_list = []
    seqs = []
    next_seqs = []
    sampled_seqs = []
    next_scores_len = 0
    cur_it = 0
    n_sequences = 60000

    with tqdm(total=n_sequences, unit="step") as pbar:
        for _, h in full_sequences.items():
            seqt = h.copy()

            if cur_it == n_sequences:
                if len(seqs):
                    states, next_states, next_scores = process_seqs(model_D_sasrec, seqs, next_seqs)

                    states_list.append(states)
                    next_states_list.append(next_states)
                    next_scores_list.append(next_scores)

                break

            if len(seqt) < 4:
                cur_it += 1
                pbar.update(1)
                pbar.set_postfix({"next_scores_len": f"{next_scores_len}"})
                continue

            idxes = random.sample(range(3, len(seqt)), min(len(seqt) - 3, samples_per_user))
            # idxes = np.random.randint(len(seqt), size=min(len(seqt), samples_per_user))
            actions += [seqt[idx][0] for idx in idxes]
            ratings += [float(seqt[idx][1]) for idx in idxes]
            seqt = list(list(zip(*seqt))[0])
            state_seqs = [seqt[max(idx - n, 0):idx] for idx in idxes]

            assert len(state_seqs) == len(idxes), "Mismatch in state_seqs after padding"
            assert len(actions[-len(state_seqs):]) == len(state_seqs), "Mismatch in actions and state_seqs"

            actions_neg.append(np.random.choice(all_actions[~np.isin(all_actions, seqt)],
                                                len(state_seqs) * n_neg_samples,
                                                replace=False).reshape(-1, n_neg_samples))


            state_seqs = [np.pad(seq, (n - len(seq), 0), mode='constant', constant_values=model_D_sasrec.pad_token).tolist() for seq in state_seqs]
            
            next_state_seqs = [state_seq[1:] + [action] for state_seq, action in zip(state_seqs, actions[-len(state_seqs):])]
            seqs += state_seqs
            next_seqs += next_state_seqs
            sampled_seqs += state_seqs

            if len(seqs) > bs:
                states, next_states, next_scores = process_seqs(model_D_sasrec, seqs, next_seqs)
                
                states_list.append(states)
                next_states_list.append(next_states)
                next_scores_list.append(next_scores)
                
                next_scores_len += next_scores.shape[0]

                seqs = []
                next_seqs = []

            cur_it += 1
            pbar.update(1)
            pbar.set_postfix({"next_scores_len": f"{next_scores_len}"})


    return np.concatenate(states_list, axis=0, dtype=np.float32),\
           np.concatenate(next_states_list, axis=0, dtype=np.float32),\
           np.concatenate(next_scores_list, axis=0, dtype=np.float16),\
           actions,\
           ratings,\
           sampled_seqs,\
           np.concatenate(actions_neg, axis=0),\
           full_sequences


def process_seq(seqt, model_D_sasrec, device):
    action, seq, rating = seqt
    
    score = None

    with torch.no_grad():
        state = model_D_sasrec.state_batch(torch.tensor(seq, device=device, dtype=torch.long).unsqueeze(0)).squeeze(0)
        state = state.detach().cpu().numpy()

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
        if len(seqt) < 3:
            continue
            
        seqt = list(zip(*seqt[-n-1:-1]))[0]

        if len(seqt) < n + 1:
            seqt = np.pad(seqt, (n + 1 - len(seqt), 0), mode='constant', constant_values=model_D_sasrec.pad_token)

        seqs.append(seqt)

        actions.append(at)
        ratings.append(rating)

    states = model_D_sasrec.state_batch(torch.LongTensor(seqs).to(device)).cpu().numpy()
    scores = model_D_sasrec.score_batch(torch.LongTensor(seqs).to(device)).cpu().numpy()

    return states, scores, actions, ratings, seqs