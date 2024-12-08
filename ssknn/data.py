from numba import njit
from numba.typed import List

from random import seed as set_seed
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from polara import get_movielens_data
from polara.preprocessing.dataframes import reindex, leave_one_out

from source.dataprep.dataprep import transform_indices
import logging

def get_dataset(validation_size=1024, test_size=5000, verbose=False, path=None, splitting='temporal_full', q=0.8):
    if type(path) == pd.core.frame.DataFrame: 
        mldata = path
    else:
        mldata = get_movielens_data(include_time=True).rename(columns={'movieid': 'itemid'})
        

    if splitting == 'full':

        # айтемы, появившимися после q
        test_timepoint = mldata['timestamp'].quantile(
            q=q, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        
        print(test_data_.nunique())
        
        train_data_ = mldata.query(
            'timestamp < @test_timepoint') # убрал userid not in @test_data_.userid.unique() and 
        training_, data_index1 = transform_indices(train_data_.copy(), 'userid', 'itemid')
        
        testset_ = reindex(test_data_, data_index1['items'])
        
        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target='timestamp', sample_top=True, random_state=0
        )
        
        testset_valid, data_index = transform_indices(testset_valid_.copy(), 'userid', 'itemid')
        
        testset_ = reindex(testset_, data_index['items'])
        holdout_valid = reindex(holdout_valid_, data_index['items'])
        
        print(testset_.nunique())
        
        # убираем из полного датасета интеракшены с айтемами, появившимися после q
        testset_valid, holdout_valid = leave_one_out(
        mldata, target='timestamp', sample_top=True, random_state=0
        )
        # training_valid_, holdout_valid_ = leave_one_out(
        #     training, target='timestamp', sample_top=True, random_state=0
        # )
        
        testset_valid = reindex(testset_valid, data_index1['items'])
        holdout_valid = reindex(holdout_valid, data_index1['items'])
        
        testset_valid = reindex(testset_valid, data_index['items'])
        holdout_valid = reindex(holdout_valid, data_index['items'])
        
        testset_ = testset_valid.copy()
        training = testset_valid.copy()

        userid = data_index['users'].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid[userid].unique(),
                holdout_valid[userid].unique()
            )
        )
        testset_valid = (
            testset_valid
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_valid = (
            holdout_valid
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
                
        holdout_ = holdout_valid.copy() # просто что то поставил как холдаут
    
    elif splitting == 'temporal_full':

        test_timepoint = mldata['timestamp'].quantile(
            q=q, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            'timestamp < @test_timepoint') # убрал userid not in @test_data_.userid.unique() and 
        training_, data_index = transform_indices(train_data_.copy(), 'userid', 'itemid')

        testset_ = reindex(test_data_, data_index['items'])

        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target='timestamp', sample_top=True, random_state=0
        )
        
        testset_valid, data_index = transform_indices(testset_valid_.copy(), 'userid', 'itemid')
        
        testset_ = reindex(testset_, data_index['items'])
        holdout_valid = reindex(holdout_valid_, data_index['items'])
        holdout_valid = reindex(holdout_valid, data_index['users']).sort_values(['userid'])
        if verbose:
            print(testset_.nunique())

        training = testset_valid.copy()


        # validation_size = 1024
        validation_users = np.intersect1d(holdout_valid['userid'].unique(), testset_valid['userid'].unique())
        if validation_size < len(validation_users):
            validation_users  = np.random.choice(validation_users, size=validation_size, replace=False)
        testset_valid = testset_valid[testset_valid['userid'].isin(validation_users)].sort_values(by=['userid', 'timestamp'])
        holdout_valid = holdout_valid[holdout_valid['userid'].isin(validation_users)]

        testset_, holdout_ = leave_one_out(
            testset_, target='timestamp', sample_top=True, random_state=0
        )

        # test_size = 5000
        test_users = np.intersect1d(holdout_['userid'].unique(), testset_['userid'].unique())
        if test_size < len(test_users):
            test_users  = np.random.choice(test_users, size=test_size, replace=False)
        testset_ = testset_[testset_['userid'].isin(test_users)].sort_values(by=['userid', 'timestamp'])
        holdout_ = holdout_[holdout_['userid'].isin(test_users)].sort_values(['userid'])
        
        # holdout_ = testset_.copy() # просто что то поставил как хзолдаут, нам метрики не нужны для теста

    elif splitting == 'temporal_full_with_history':

        test_timepoint = mldata['timestamp'].quantile(
            q=q, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            'timestamp < @test_timepoint') # убрал userid not in @test_data_.userid.unique() and 
        training_, data_index = transform_indices(train_data_.copy(), 'userid', 'itemid')

        testset_ = reindex(test_data_, data_index['items'])
        
        ###
        test_user_idx = data_index['users'].get_indexer(testset_['userid'])
        is_new_user = test_user_idx == -1
        if is_new_user.any(): # track unseen users - to be used in warm-start regime
            new_user_idx, data_index['new_users'] = pd.factorize(testset_.loc[is_new_user, 'userid'])
            # ensure no intersection with train users index
            test_user_idx[is_new_user] = new_user_idx + len(data_index['users'])
        # assign new user index
        testset_.loc[:, 'userid'] = test_user_idx
        ###
        
        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target='timestamp', sample_top=True, random_state=0
        )
        
        testset_valid, data_index = transform_indices(testset_valid_.copy(), 'userid', 'itemid')
        
        testset_ = reindex(testset_, data_index['items'])
        ###
        test_user_idx = data_index['users'].get_indexer(testset_['userid'])
        is_new_user = test_user_idx == -1
        if is_new_user.any(): # track unseen users - to be used in warm-start regime
            new_user_idx, data_index['new_users'] = pd.factorize(testset_.loc[is_new_user, 'userid'])
            # ensure no intersection with train users index
            test_user_idx[is_new_user] = new_user_idx + len(data_index['users'])
        # assign new user index
        testset_.loc[:, 'userid'] = test_user_idx
        ###
        
        holdout_valid = reindex(holdout_valid_, data_index['items'])
        holdout_valid = reindex(holdout_valid, data_index['users'])
        
        if verbose:
            print(testset_.nunique())

        training = testset_valid.copy()

        # userid = data_index['users'].name
        # test_users = pd.Index(
        #     # ensure test users are the same across testing data
        #     np.intersect1d(
        #         testset_valid[userid].unique(),
        #         holdout_valid[userid].unique()
        #     )
        # )
        testset_valid = (
            testset_valid
            # reindex warm-start users for convenience
            # .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            # .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_valid = (
            holdout_valid
            # reindex warm-start users for convenience
            # .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            # .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        
        holdout_ = testset_.copy() # просто что то поставил как хзолдаут, нам метрики не нужны для теста
    
    elif splitting == 'temporal':
        test_timepoint = mldata['timestamp'].quantile(
            q=0.95, interpolation='nearest'
        )
        test_data_ = mldata.query('timestamp >= @test_timepoint')
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            'userid not in @test_data_.userid.unique() and timestamp < @test_timepoint'
        )
        training, data_index = transform_indices(train_data_.copy(), 'userid', 'itemid')

        test_data = reindex(test_data_, data_index['items'])
        if verbose:
            print(test_data.nunique())
        testset_, holdout_ = leave_one_out(
            test_data, target='timestamp', sample_top=True, random_state=0
        )
        testset_valid_, holdout_valid_ = leave_one_out(
            testset_, target='timestamp', sample_top=True, random_state=0
        )
        
        userid = data_index['users'].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid_[userid].unique(),
                holdout_valid_[userid].unique()
            )
        )
        testset_valid = (
            testset_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_valid = (
            holdout_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
    
        testset_ = (
            testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_ = (
            holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )

    elif splitting == 'leave-one-out':
        mldata, data_index = transform_indices(mldata.copy(), 'userid', 'itemid')
        training, holdout_ = leave_one_out(
        mldata, target='timestamp', sample_top=True, random_state=0
        )
        training_valid_, holdout_valid_ = leave_one_out(
            training, target='timestamp', sample_top=True, random_state=0
        )

        testset_valid_ = training_valid_.copy()
        testset_ = training.copy()
        training = training_valid_.copy()

        userid = data_index['users'].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid_[userid].unique(),
                holdout_valid_[userid].unique()
            )
        )
        testset_valid = (
            testset_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_valid = (
            holdout_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
    
        testset_ = (
            testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )
        holdout_ = (
            holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f'{userid} >= 0')
            .sort_values('userid')
        )

    else:
        raise ValueError
    
    if verbose:
        print(testset_valid.nunique())
        print(holdout_valid.shape)
    # assert holdout_valid.set_index('userid')['timestamp'].ge(
    #     testset_valid
    #     .groupby('userid')
    #     ['timestamp'].max()
    # ).all()

    data_description = dict(
        users = data_index['users'].name,
        items = data_index['items'].name,
        order = 'timestamp',
        n_users = len(data_index['users']),
        n_items = len(data_index['items']),
    )

    if verbose:
        print(data_description)

    return training, data_description, testset_valid, testset_, holdout_valid, holdout_

def data_to_sequences(data, data_description):
    userid = data_description['users']
    itemid = data_description['items']
    sequences = (
        data.sort_values([userid, data_description['order']])
        .groupby(userid, sort=False)[itemid].apply(list)
    )
    return sequences

def data_to_sequences_rating(data, data_description, n_neg_samples=-1):
    logger = logging.getLogger("fqe")
    userid = data_description['users']
    itemid = data_description['items']
    order = data_description['order']

    sequences = (
        data.sort_values([userid, order])
        .groupby(userid, sort=False)
        .apply(lambda x: list(zip(list(x[itemid]), list(x['rating']))))
    )
    # sequences = (
    #     data.sort_values([userid, data_description['order']])
    #     .groupby(userid, sort=False)[itemid].apply(list)
    # )
    # sequences = sequences.apply(lambda x: list(zip(x, [1] * len(x)))) #!!!!!!
    logger.info(sequences.shape[0])
    
    logger.info("sequences processed!")
    if n_neg_samples > 0:
        n_items = data_description['n_items']
        sequences = sequences.apply(lambda x: sample_f(x, n_items, n_neg_samples))
        
    logger.info("negative sampling processed!")

    return sequences


if __name__ == '__main__':
    get_dataset()
