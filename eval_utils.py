import numpy as np
import torch

from utils import topn_recommendations, downvote_seen_items
from data import data_to_sequences
from tqdm import tqdm

def get_test_scores(model, data_description, testset_, holdout_, device):
    """
    Computes evaluation scores for a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_description (dict): Description of the dataset including user and item fields.
        testset_ (pd.DataFrame): The test set containing user-item interactions.
        holdout_ (pd.DataFrame): The holdout set for evaluation.
        device (torch.device): Device to perform computations on.

    Returns:
        dict: Evaluation metrics including HR, MRR, NDCG, and Coverage.
    """
    sasrec_scores = sasrec_model_scoring(model, testset_, data_description, device)
    downvote_seen_items(sasrec_scores, testset_, data_description)

    sasrec_recs = topn_recommendations(sasrec_scores, topn=50)
    test_scores = model_evaluate(sasrec_recs, holdout_, data_description)
    return test_scores

def sasrec_model_scoring(params, data, data_description, device, test_batch_size=512):
    """
    Computes model scores for sequences in the test dataset.

    Args:
        params (torch.nn.Module): The SASRec model parameters.
        data (pd.DataFrame): The test dataset.
        data_description (dict): Description of the dataset including user and item fields.
        device (torch.device): Device to perform computations on.
        test_batch_size (int, optional): Batch size for scoring. Defaults to 512.

    Returns:
        np.ndarray: Predicted scores for all users in the test dataset.
    """
    model = params
    model.eval()
    test_sequences = data_to_sequences(data, data_description)

    scores = []
    for j in range(0, len(test_sequences), test_batch_size):
        with torch.no_grad():
            seqs = test_sequences.iloc[j:j+test_batch_size].tolist()
            maxlen = max([len(i) for i in seqs])
            seqs_padded = [np.pad(arr, (maxlen - len(arr), 0), mode='constant', constant_values=model.pad_token) for arr in seqs]
            seqs_padded = torch.tensor(seqs_padded, device=device, dtype=torch.long)[:, -model.pos_emb.num_embeddings:] 

            predictions = model.score_batch(seqs_padded).detach().cpu()

        scores.append(predictions)
    return np.concatenate(scores, axis=0)

def calculate_topn_metrics(recommended_items, holdout_items, n_items, n_test_users, topn):
    """
    Calculates evaluation metrics such as HR, MRR, NDCG, and Coverage.

    Args:
        recommended_items (np.ndarray): Array of recommended items for each user.
        holdout_items (np.ndarray): Ground truth items for each user.
        n_items (int): Total number of items.
        n_test_users (int): Number of users in the test dataset.
        topn (int): Number of top recommendations to consider.

    Returns:
        dict: Evaluation metrics for the top-N recommendations.
    """
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)

    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))

    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
   
    # NDCG calculation
    ndcg = np.sum(1 / np.log2(hit_rank + 1.)) / n_test_users

    # Coverage calculation
    cov = np.unique(recommended_items[:, :topn]).size / n_items

    return {'hr': hr, 'mrr': mrr, 'ndcg': ndcg, 'cov': cov}

def model_evaluate(recommended_items, holdout, holdout_description, topn_list=(1, 5, 10, 20, 50)):
    """
    Evaluates a model's recommendations against the ground truth holdout set.

    Args:
        recommended_items (np.ndarray): Array of recommended items for each user.
        holdout (pd.DataFrame): Ground truth holdout set.
        holdout_description (dict): Description of the holdout set, including item field.
        topn_list (tuple, optional): List of top-N values to compute metrics for. Defaults to (1, 5, 10, 20, 50).

    Returns:
        dict: Evaluation metrics for the provided top-N values.
    """
    n_items = holdout_description['n_items']
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    n_test_users = recommended_items.shape[0]
    assert recommended_items.shape[0] == len(holdout_items)

    metrics = {}
    
    for topn in topn_list:
        new_metrics = {f'{key}@{topn}': value \
        for key, value in calculate_topn_metrics(recommended_items, holdout_items, n_items, n_test_users, topn).items()}
        metrics.update(new_metrics)

    return metrics
