import numpy as np
import pandas as pd


def downvote_seen_items(scores, data, data_description):
    assert isinstance(scores, np.ndarray), 'Scores must be a dense numpy array!'
    itemid = data_description['items']
    userid = data_description['users']
    # get indices of observed data, corresponding to scores array
    # we need to provide correct mapping of rows in scores array into
    # the corresponding user index (which is assumed to be sorted)
    row_idx, test_users = pd.factorize(data[userid], sort=True)
    assert len(test_users) == scores.shape[0]
    col_idx = data[itemid].values
    # downvote scores at the corresponding positions
    scores[row_idx, col_idx] = scores.min() - 1


def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


def model_evaluate(recommended_items, holdout, holdout_description, topn=10):
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    assert recommended_items.shape[0] == len(holdout_items)
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))
    # MRR calculation
    n_test_users = recommended_items.shape[0]
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1. / hit_rank) / n_test_users
    ndcg = np.sum(1. / np.log2(hit_rank + 1)) / n_test_users
    # coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
    return {'hr':hr, 'mrr':mrr, 'ndcg':ndcg, 'cov':cov}


def build_evaluate_model(Model, model_config, data_dict, data_description):
    '''
    Builds the model and calculates metric using the holdout set.
    Args:
        Model (model class): The model class to train
        model config (dict): A dictionary with model hyperparameters
        data_dict (dict): A dictionary containing data with the following keys:
            - 'train' (pd.DataFrame): The input dataframe containing the train user-item interactions.
            - 'test' (pd.DataFrame): The input dataframe containing the test user-item interactions.
            - 'holdout' (pd.DataFrame): The input dataframe containing the holdout to measure the quality of recommendations.
        data_description (dict): A dictionary containing the data description with the following keys:
            - 'n_users' (int): The total number of unique users in the data.
            - 'n_items' (int): The total number of unique items in the data.
            - 'users' (str): The name of the column in the dataframe containing the user ids.
            - 'items' (str): The name of the column in the dataframe containing the item ids.
            - 'feedback' (str): The name of the column in the dataframe containing the user-item interaction feedback.
            - 'timestamp' (str): The name of the column in the dataframe containing the timestamps of interactions.

    Returns:
        np.array: A numpy array of shape (n_test_users, n_items) containing the scores for each user.
    '''
    model = Model(model_config)
    model.build(data_dict['train'], data_description)
    preds = model.recommend(data_dict['test'], data_description)
    downvote_seen_items(preds, data_dict['test'], data_description)
    recs = topn_recommendations(preds)
    metrics = model_evaluate(recs, data_dict['holdout'], data_description)

    return metrics, model