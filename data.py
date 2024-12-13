from numba import njit
from numba.typed import List

from random import seed as set_seed
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from polara import get_movielens_data
from polara.preprocessing.dataframes import reindex, leave_one_out

from sampler import prime_sampler_state, sample_unseen
from utils import transform_indices
import logging

def get_dataset(verbose=False, path='None', splitting='temporal_full', q=0.8):
    """
    Prepares and splits the dataset for training, validation, and testing.

    Args:
        verbose (bool): If True, prints additional information during processing.
        path (str): Path to the dataset file. If 'None', loads the Movielens dataset.
        splitting (str): Method for splitting the dataset. Options include 'full', 'temporal_full', 'leave-one-out', etc.
        q (float): Quantile used for temporal splitting.

    Returns:
        tuple: Training set, data description, validation set, test set, validation holdout set, and test holdout set.
    """
    if path != 'None':
        mldata = path
    else:
        mldata = get_movielens_data(include_time=True, local_file='ml-1m.zip').rename(columns={'movieid': 'itemid'})

    if splitting == 'full':
        test_timepoint = mldata['timestamp'].quantile(q=q, interpolation='nearest')
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        train_data_ = mldata.query('timestamp < @test_timepoint')

        training_, data_index1 = transform_indices(train_data_.copy(), 'userid', 'itemid')
        testset_ = reindex(test_data_, data_index1['items'])

        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target='timestamp', sample_top=True, random_state=0
        )

        testset_valid, data_index = transform_indices(testset_valid_.copy(), 'userid', 'itemid')
        testset_ = reindex(testset_, data_index['items'])
        holdout_valid = reindex(holdout_valid_, data_index['items'])

        training = testset_valid.copy()
        holdout_ = holdout_valid.copy()

    elif splitting == 'temporal_full':
        test_timepoint = mldata['timestamp'].quantile(q=q, interpolation='nearest')
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        train_data_ = mldata.query('timestamp < @test_timepoint')

        training_, data_index = transform_indices(train_data_.copy(), 'userid', 'itemid')
        testset_ = reindex(test_data_, data_index['items'])
        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target='timestamp', sample_top=True, random_state=0
        )

        testset_valid, data_index = transform_indices(testset_valid_.copy(), 'userid', 'itemid')
        holdout_valid = reindex(holdout_valid_, data_index['items'])

        training = testset_valid.copy()
        holdout_ = testset_.copy()

    else:
        raise ValueError("Invalid splitting method")

    data_description = dict(
        users=data_index['users'].name,
        items=data_index['items'].name,
        order='timestamp',
        n_users=len(data_index['users']),
        n_items=len(data_index['items']),
    )

    return training, data_description, testset_valid, testset_, holdout_valid, holdout_

def no_sample(user_items, maxlen, pad_token):
    """
    Prepares sequences without negative sampling.

    Args:
        user_items (list): List of user interactions.
        maxlen (int): Maximum length of the sequence.
        pad_token (int): Padding token used for sequences.

    Returns:
        tuple: Sequence, positive interactions, and negative samples.
    """
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.empty((maxlen, 1))

    n_user_items = min(len(user_items) - 1, maxlen)

    seq[-n_user_items:] = user_items[-n_user_items-1:-1]
    pos[-n_user_items:] = user_items[-n_user_items:]

    return seq, pos, neg

def sample_with_rep(user_items, maxlen, pad_token, n_neg_samples, itemnum, random_state):
    """
    Prepares sequences with replacement for negative sampling.

    Args:
        user_items (list): List of user interactions.
        maxlen (int): Maximum length of the sequence.
        pad_token (int): Padding token used for sequences.
        n_neg_samples (int): Number of negative samples per sequence.
        itemnum (int): Total number of items.
        random_state (np.random.RandomState): Random state for reproducibility.

    Returns:
        tuple: Sequence, positive interactions, and negative samples.
    """
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.full((n_neg_samples, maxlen), pad_token, dtype=np.int32)

    n_user_items = min(len(user_items) - 1, maxlen)

    seq[-n_user_items:] = user_items[-n_user_items-1:-1]
    pos[-n_user_items:] = user_items[-n_user_items:]
    neg[:, -n_user_items:] = random_state.randint(0, itemnum, (n_neg_samples, n_user_items))

    return seq, pos, neg

@njit
def sample_without_rep(user_items, maxlen, pad_token, n_neg_samples, itemnum, seed):
    """
    Prepares sequences without replacement for negative sampling.

    Args:
        user_items (list): List of user interactions.
        maxlen (int): Maximum length of the sequence.
        pad_token (int): Padding token used for sequences.
        n_neg_samples (int): Number of negative samples per sequence.
        itemnum (int): Total number of items.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Sequence, positive interactions, and negative samples.
    """
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.full((maxlen, n_neg_samples), pad_token, dtype=np.int32)

    hist_items_counter = 1
    nxt = user_items[-1]
    idx = maxlen - 1

    set_seed(seed)

    ts_ = list(set(user_items))

    for i in user_items[-2::-1]:
        seq[idx] = i
        pos[idx] = nxt

        state = prime_sampler_state(itemnum, ts_)
        remaining = itemnum - len(ts_)
        
        sample_unseen(n_neg_samples, state, remaining, neg[idx])

        nxt = i
        idx -= 1
        hist_items_counter += 1
        if idx == -1:
            break
        
    neg = np.swapaxes(neg, 0, 1)
    return seq, pos, neg

class SequentialDataset(Dataset):
    """
    PyTorch Dataset for sequential recommendation tasks.

    Args:
        user_train (dict): Dictionary mapping user IDs to their interactions.
        usernum (int): Number of users.
        itemnum (int): Number of items.
        maxlen (int): Maximum sequence length.
        seed (int): Random seed for reproducibility.
        n_neg_samples (int): Number of negative samples per sequence.
        sampling (str): Sampling method ('with_rep', 'without_rep', 'dross', 'no_sampling').
        pad_token (int): Padding token used for sequences.
    """
    def __init__(self, user_train, usernum, itemnum, maxlen, seed, n_neg_samples=1, sampling='without_rep', pad_token=None):
        super().__init__()
        self.user_train = user_train

        self.valid_users = [user for user in range(usernum) if len(user_train.get(user, [])) > 1]

        self.usernum = len(self.valid_users)

        self.itemnum = itemnum
        self.maxlen = maxlen
        self.seed = seed
        self.n_neg_samples = n_neg_samples
        self.sampling = sampling
        
        if self.sampling == 'dross':
            self.all_items = np.arange(self.itemnum, dtype=np.int32)

        self.pad_token = pad_token

        self.random_state = np.random.RandomState(self.seed)

    def __len__(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: Number of users.
        """
        return self.usernum
    
    def __getitem__(self, idx):
        """
        Retrieves sequences, positive, and negative samples for a given user index.

        Args:
            idx (int): Index of the user.

        Returns:
            tuple: User ID, sequence, positive samples, and negative samples.
        """
        user = self.valid_users[idx]
        user_items = List()
        [user_items.append(x) for x in self.user_train[user]]

        if self.sampling == 'with_rep':
            seq, pos, neg = sample_with_rep(user_items, self.maxlen, self.pad_token,
                                            self.n_neg_samples, self.itemnum,
                                            self.random_state)
        elif self.sampling == 'without_rep':
            seq, pos, neg = sample_without_rep(user_items, self.maxlen, self.pad_token,
                                               self.n_neg_samples, self.itemnum,
                                               self.random_state.randint(np.iinfo(int).min, np.iinfo(int).max))
        elif self.sampling == 'dross':
            seq, pos, neg = sample_dross(self.all_items, user_items, self.maxlen, self.pad_token,
                                         self.n_neg_samples,
                                         self.random_state)
        elif self.sampling == 'no_sampling':
            seq, pos, neg = no_sample(user_items, self.maxlen, self.pad_token)
        else:
            raise NotImplementedError()

        return user, seq, pos, neg 
