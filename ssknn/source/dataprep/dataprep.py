from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from polara.preprocessing.dataframes import leave_one_out, reindex


def leave_last_out(data, userid='userid', timeid='timestamp'):
    data_sorted = data.sort_values('timestamp')
    holdout = data_sorted.drop_duplicates(
        subset=['userid'], keep='last'
    ) # split the last item from each user's history
    remaining = data.drop(holdout.index) # store the remaining data - will be our training
    return remaining, holdout


def transform_indices(data, users, items):
    '''
    Reindex columns that correspond to users and items.
    New index is contiguous starting from 0.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data to be reindexed.
    users : str
        The name of the column in `data` that contains user IDs.
    items : str
        The name of the column in `data` that contains item IDs.

    Returns
    -------
    pandas.DataFrame, dict
        The reindexed data and a dictionary with mapping between original IDs and the new numeric IDs.
        The keys of the dictionary are 'users' and 'items'.
        The values of the dictionary are pandas Index objects.

    Examples
    --------
    >>> data = pd.DataFrame({'customers': ['A', 'B', 'C'], 'products': ['X', 'Y', 'Z'], 'rating': [1, 2, 3]})
    >>> data_reindexed, data_index = transform_indices(data, 'customers', 'products')
    >>> data_reindexed
       users  items  rating
    0      0      0       1
    1      1      1       2
    2      2      2       3
    >>> data_index
    {
        'users': Index(['A', 'B', 'C'], dtype='object', name='customers'),
        'items': Index(['X', 'Y', 'Z'], dtype='object', name='products')
    }
    '''
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        new_index, data_index[entity] = to_numeric_id(data, field)
        data = data.assign(**{f'{field}': new_index}) # makes a copy of dataset!
    return data, data_index


def to_numeric_id(data, field):
    """
    This function takes in two arguments, data and field. It converts the data field
    into categorical values and creates a new contiguous index. It then creates an
    idx_map which is a renamed version of the field argument. Finally, it returns the
    idx and idx_map variables. 
    """
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def reindex_data(data, data_index, fields=None):
    '''
    Reindex provided data with the specified index mapping.
    By default, will take the name of the fields to reindex from `data_index`.
    It is also possible to specify which field to reindex by providing `fields`.
    '''
    if fields is None:
        fields = data_index.keys()
    if isinstance(fields, str): # handle single field provided as a string
        fields = [fields]
    for field in fields:
        entity_name = data_index[field].name
        new_index = data_index[field].get_indexer(data[entity_name])
        data = data.assign(**{f'{entity_name}': new_index}) # makes a copy of dataset!
    return data


def generate_interactions_matrix(data, data_description, rebase_users=False, implicit=True):
    '''
    Converts a pandas dataframe with user-item interactions into a sparse matrix representation.
    Allows reindexing user ids, which help ensure data consistency at the scoring stage
    (assumes user ids are sorted in the scoring array).
    
    Args:
        data (pandas.DataFrame): The input dataframe containing the user-item interactions.
        data_description (dict): A dictionary containing the data description with the following keys:
            - 'n_users' (int): The total number of unique users in the data.
            - 'n_items' (int): The total number of unique items in the data.
            - 'users' (str): The name of the column in the dataframe containing the user ids.
            - 'items' (str): The name of the column in the dataframe containing the item ids.
            - 'feedback' (str): The name of the column in the dataframe containing the user-item interaction feedback.
        rebase_users (bool, optional): Whether to reindex the user ids to make contiguous index starting from 0. Defaults to False.

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix of shape (n_users, n_items) containing the user-item interactions.
    '''
        
    n_users = data_description['n_users']
    n_items = data_description['n_items']
    # get indices of observed data
    user_idx = data[data_description['users']].values
    if rebase_users: # handle non-contiguous index of test users
        # This ensures that all user ids are contiguous and start from 0,
        # which helps ensure data consistency at the scoring stage.
        user_idx, user_index = pd.factorize(user_idx, sort=True)
        n_users = len(user_index)
    item_idx = data[data_description['items']].values
    if implicit:
        feedback = np.ones(len(item_idx))
    else:
        feedback = data[data_description['feedback']].values
    # construct rating matrix
    return csr_matrix((feedback, (user_idx, item_idx)), shape=(n_users, n_items))


def generate_sequential_matrix(data, data_description, rebase_users=False):
    '''
    Converts a pandas dataframe with user-item interactions into a sparse matrix representation.
    Allows reindexing user ids, which help ensure data consistency at the scoring stage
    (assumes user ids are sorted in the scoring array).

    Args:
        data (pandas.DataFrame): The input dataframe containing the user-item interactions.
        data_description (dict): A dictionary containing the data description with the following keys:
            - 'n_users' (int): The total number of unique users in the data.
            - 'n_items' (int): The total number of unique items in the data.
            - 'users' (str): The name of the column in the dataframe containing the user ids.
            - 'items' (str): The name of the column in the dataframe containing the item ids.
            - 'feedback' (str): The name of the column in the dataframe containing the user-item interaction feedback.
            - 'timestamp' (str): The name of the column in the dataframe containing the user-item interaction timestamp.
        rebase_users (bool, optional): Whether to reindex the user ids to make contiguous index starting from 0. Defaults to False.

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix of shape (n_users, n_items) containing the user-item interactions with reciprocal weighting.
    '''

    data_sorted = data.sort_values(by=[data_description['timestamp']], ascending=False)
    data_sorted['reciprocal_rank'] = (1.0 / (data_sorted.groupby(data_description['users']).cumcount() + 1))

    n_users = data_description['n_users']
    n_items = data_description['n_items']
    # get indices of observed data
    user_idx = data_sorted[data_description['users']].values
    if rebase_users: # handle non-contiguous index of test users
        # This ensures that all user ids are contiguous and start from 0,
        # which helps ensure data consistency at the scoring stage.
        user_idx, user_index = pd.factorize(user_idx, sort=True)
        n_users = len(user_index)
    item_idx = data_sorted[data_description['items']].values
    ranks = data_sorted['reciprocal_rank'].values
    # construct the matrix
    return csr_matrix((ranks, (user_idx, item_idx)), shape=(n_users, n_items), dtype='f8')

def generate_histories(data, data_description):
    histories = list(
        data
        .sort_values(by=[data_description['users'], data_description['timestamp']])
        .groupby(data_description['users'])[data_description['items']]
        .apply(list)
        .values
        )
    return histories

def generate_tensor(data, data_description, rebase_users=False):
    userid = data_description["users"]
    itemid = data_description["items"]
    positions = data_description["positions"]
    
    df = data.copy()
    
    if rebase_users:
        user_idx, user_index = pd.factorize(
            df[data_description['users']].values,
            sort=True
            )
        df['factorized'] = user_idx
        idx = df[['factorized', itemid, positions]].values
    else:
        idx = data[[userid, itemid, positions]].values.T
    val = np.ones(idx.shape[1], dtype='f8')
    return idx, val

def verify_time_split(before, after, target_field='userid', timeid='timestamp'):
    '''
    Check that items from `after` dataframe have later timestamps than
    any corresponding item from the `before` dataframe. Compare w.r.t target_field.
    Usage example: assert that for any user, the holdout items are the most recent ones.
    '''
    before_ts = before.groupby(target_field)[timeid].max()
    after_ts = after.groupby(target_field)[timeid].min()
    assert (
        before_ts
        .reindex(after_ts.index)
        .combine(after_ts, lambda x, y: True if x!=x else x <= y)
    ).all()
    
def split_data_global_timepoint(data, data_description, quantile=0.95):
    # define the timepoint corresponding to the 95% percentile
    test_timepoint = data['timestamp'].quantile(
        q=quantile, interpolation='nearest'
    )

    # users with interaction after timepoint go to test
    _test_data_ = data.query(f'{data_description["timestamp"]} >= @test_timepoint')
    test_users = _test_data_[data_description['users']].unique()
    test_data_ = data.query(
        f'{data_description["users"]} in @test_users'
    )
    # interaction before timepoint go to train,
    # also hiding the interactions of test users
    # this ensures the warm-start strategy
    train_data_ = data.query(
        f'{data_description["users"]} not in @test_data_.{data_description["users"]}.unique() and {data_description["timestamp"]} < @test_timepoint'
    )
    
    # transform user and item ids for convenience, reindex test data
    training, data_index = transform_indices(train_data_.copy(), data_description['users'], data_description['items'])

    # reindex items in test set, if item was not in train, assign -1 as itemid
    test_data = reindex(test_data_, data_index['items'], filter_invalid=False)
    # the items that were not in the training set have itemid -1
    # let's drop the items with itemid -1 and all consequtive interactions
    test_data = test_data.sort_values(by=[data_description['users'], data_description['timestamp']])
    mask = test_data.groupby(data_description['users']).cummin()[data_description['items']] == -1
    test_data_truncated = test_data[~mask]

    filtered = test_data_truncated[test_data_truncated[data_description['timestamp']] >= test_timepoint]
    interaction_counts = filtered.groupby(data_description['users']).size()
    test_users = interaction_counts[interaction_counts >= 2].index.tolist()

    test_prepared = test_data_truncated[test_data_truncated[data_description['users']].isin(test_users)]
    
    # last interaction for test holdout
    # second-to-last for validation holdout
    testset_, holdout_ = leave_one_out(
        test_prepared, target='timestamp', sample_top=True, random_state=0
    )
    testset_valid_, holdout_valid_ = leave_one_out(
        testset_, target='timestamp', sample_top=True, random_state=0
    )

    # assert the users in testset are the same as in holdout
    test_users = np.intersect1d(testset_valid_.userid.unique(), holdout_valid_.userid.unique())
    testset_valid = testset_valid_.query(f'{data_description["users"]} in @test_users').sort_values(data_description['users'])
    holdout_valid = holdout_valid_.query(f'{data_description["users"]} in @test_users').sort_values(data_description['users'])

    # assert the users in testset_valid are the same as in holdout_valid
    test_users_final = np.intersect1d(testset_valid_.userid.unique(), holdout_valid_.userid.unique())
    testset = testset_.query(f'{data_description["users"]} in @test_users_final').sort_values(data_description['users'])
    holdout = holdout_.query(f'{data_description["users"]} in @test_users_final').sort_values(data_description['users'])
    
    return training, testset_valid, holdout_valid, testset, holdout, data_index, data_description