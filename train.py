from time import time
from functools import reduce
import uuid
from tqdm import tqdm
import os

from clearml import Task, Logger
from omegaconf import OmegaConf
import hydra

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import SASRec
from data import get_dataset, data_to_sequences, SequentialDataset
from utils import topn_recommendations, downvote_seen_items
from eval_utils import model_evaluate, sasrec_model_scoring, get_test_scores

import logging


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config):
    """
    The main entry point for training the SASRec model.

    Initializes the ClearML task, sets up configurations, and prepares the dataset.
    Builds and trains the model based on the provided configuration.
    """
    print(OmegaConf.to_yaml(config))
    
    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)
    
    if hasattr(config, 'project_name'):
        Task.set_random_seed(config.trainer_params.seed)
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None
        
        
    base_config = dict(
        num_epochs = config.trainer_params.num_epochs,
        maxlen = config.model_params.maxlen,
        hidden_units = config.model_params.hidden_units,
        dropout_rate = config.model_params.dropout_rate,
        num_blocks = config.model_params.num_blocks,
        num_heads = config.model_params.num_heads,
        batch_size = config.dataloader.batch_size,
        learning_rate = config.trainer_params.learning_rate,
        fwd_type = config.model_params.fwd_type,
        l2_emb = 0,
        n_neg_samples = config.dataloader.n_neg_samples,
        manual_seed = config.trainer_params.seed,
        sampler_seed = config.trainer_params.seed,
        sampling = config.model_params.sampling,
        patience = config.trainer_params.patience,
        skip_epochs = config.trainer_params.skip_epochs
    )

    device = 'cuda'
    training, data_description, testset_valid, testset_, holdout_valid, holdout_ = \
        get_dataset(path=config.data_path, splitting=config.splitting)

    if task:
        log = Logger.current_logger()
    else:
        log = None

    userid = data_description['users']
    validation_size = 5000
    validation_users = testset_valid[userid].unique()
    if validation_size < len(validation_users):
        validation_users  = np.random.choice(validation_users, size=validation_size, replace=False)
        testset_valid = testset_valid[testset_valid[userid].isin(validation_users)]
        holdout_valid = holdout_valid[holdout_valid[userid].isin(validation_users)]
    
    model = \
        build_sasrec_model(base_config, training, data_description,
                           testset_valid=testset_valid, holdout_valid=holdout_valid, device=device,
                           task=task, log=log)
        

def set_worker_random_state(id):
    """
    Sets the random state for a worker in a DataLoader.

    Ensures reproducibility by initializing the worker's dataset seed
    with a unique value based on the worker's ID.
    """
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.seed = dataset.seed + id
    dataset.random_state = np.random.RandomState(dataset.seed)


def prepare_sasrec_model(config, data, data_description, device, item_emb_svd=None):
    """
    Prepares the SASRec model, dataset sampler, and optimizer for training.

    Args:
        data: Training dataset.
        data_description: Metadata about the dataset, such as the number of users and items.
        item_emb_svd: Pre-trained item embeddings (optional).

    Returns:
        tuple: The SASRec model, a DataLoader for sampling, the number of batches, and the optimizer.
    """
    logger = logging.getLogger("fqe")
    n_users = data_description['n_users']
    n_items = data_description['n_items']

    model = SASRec(n_items, config, item_emb_svd).to(device)

    logger.info("model created!")
    train_sequences = data_to_sequences(data, data_description)
    # train_sequences.to_csv('./mdp/train_sequences.csv')
    # train_sequences = pd.read_csv("./mdp/train_sequences.csv", index_col=0, squeeze=True).rename(None).apply(eval)
    logger.info("train_sequences generated!")

    sampler = \
        DataLoader(SequentialDataset(train_sequences, n_users, n_items,
            maxlen = config['maxlen'],
            seed = config['sampler_seed'],
            n_neg_samples = config['n_neg_samples'],
            pad_token = model.pad_token,
            sampling = config['sampling']
            ),
        batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=10,
        worker_init_fn=set_worker_random_state, persistent_workers=True, drop_last=True)

    n_batches = len(train_sequences) // config['batch_size']

    optimizer = \
    torch.optim.Adam(model.parameters(),
        lr = config['learning_rate'],
        betas = (0.9, 0.98))

    return model, sampler, n_batches, optimizer


def train_sasrec_epoch(model, num_batch, l2_emb, sampler, optimizer, device):
    """
    Trains the SASRec model for one epoch.

    Args:
        num_batch: Number of batches to train in one epoch.
        l2_emb: Regularization parameter for embedding layers.
        sampler: DataLoader object for sampling training data.
        optimizer: Optimizer for updating the model's parameters.

    Returns:
        list: List of loss values for each batch.
    """
    model.train()
    pad_token = model.pad_token ###??????????????????
    losses = []
    for _, *seq_data in sampler:
        # convert batch data into torch tensors
        seq, pos, neg = (torch.tensor(np.array(x), device=device, dtype=torch.long) for x in seq_data)
        loss = model(seq, pos, neg)
        optimizer.zero_grad()
        if l2_emb != 0:
            for param in model.item_emb.parameters():
                loss += l2_emb * torch.norm(param)**2
        loss.backward()        
        optimizer.step()
        losses.append(loss.item())
    return losses


def build_sasrec_model(config, data, data_description, testset_valid, holdout_valid, device, task, log, item_emb_svd=None):
    """
    Builds and trains the SASRec model.

    Performs training over multiple epochs with evaluation at specified intervals.
    Includes functionality for early stopping based on validation performance.

    Args:
        data: Training dataset.
        data_description: Metadata about the dataset.
        testset_valid: Validation set used for intermediate evaluation.
        holdout_valid: Holdout set for testing the model's performance.
        task: ClearML task object for logging and monitoring.
        log: Logger instance for reporting metrics.

    Returns:
        SASRec: The trained SASRec model.
    """
    model, sampler, n_batches, optimizers = prepare_sasrec_model(config, data, data_description, device, item_emb_svd)
    losses = {}
    metrics = {}
    ndcg = {}
    best_ndcg = 0
    wait = 0

    # userid = data_description['users']
    # validation_size = 5000
    # validation_users = testset_valid[userid].unique()
    # if validation_size < len(validation_users):
    #     validation_users  = np.random.choice(validation_users, size=validation_size, replace=False)
    #     testset_valid = testset_valid[testset_valid[userid].isin(validation_users)]
    #     holdout_valid = holdout_valid[holdout_valid[userid].isin(validation_users)]

    start_time = time()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()

    checkpt_name = uuid.uuid4().hex
    if not os.path.exists('./checkpt'):
        os.mkdir('./checkpt')

    checkpt_path = os.path.join('./checkpt', f'{checkpt_name}.chkpt')

    for epoch in tqdm(range(config['num_epochs'])):
        losses[epoch] = train_sasrec_epoch(
            model, n_batches, config['l2_emb'], sampler, optimizers, device
        )
        print(f'loss={np.mean(losses[epoch])}, epoch={epoch}')
        if epoch % config['skip_epochs'] == 0:
            val_scores = sasrec_model_scoring(model, testset_valid, data_description, device)
            # scores = val_scores.copy() #
            # downvote_seen_items(val_scores, testset_valid, data_description)
            val_recs = topn_recommendations(val_scores, topn=10)
            val_metrics = model_evaluate(val_recs, holdout_valid, data_description)
            metrics[epoch] = val_metrics
            ndcg_ = val_metrics['ndcg@10']
            ndcg[epoch] = ndcg_

            print(f'Epoch {epoch}, NDCG@10: {ndcg_}')
            hr_ = val_metrics['hr@10']
            cov_ = val_metrics['cov@10']
            print(f'Epoch {epoch}, HR@10: {hr_}')
            print(f'Epoch {epoch}, Cov@10: {cov_}')

            if task and (epoch % 5 == 0):
                log.report_scalar("Loss", series='Val', iteration=epoch, value=np.mean(losses[epoch]))
                log.report_scalar("NDCG", series='Val', iteration=epoch, value=ndcg_)

            if ndcg_ > best_ndcg:
                best_ndcg = ndcg_
                torch.save(model.state_dict(), checkpt_path)
                wait = 0
            elif wait < config['patience'] // config['skip_epochs'] + 1:
                wait += 1
            else:
                break

    torch.cuda.synchronize()
    training_time_sec = time() - start_time
    full_peak_training_memory_bytes = torch.cuda.max_memory_allocated()
    peak_training_memory_bytes = torch.cuda.max_memory_allocated() - start_memory
    training_epoches = len(losses)

    model.load_state_dict(torch.load(checkpt_path))
    # os.remove(checkpt_path)

    print()
    print('Peak training memory, mb:', round(full_peak_training_memory_bytes/ 1024. / 1024., 2))
    print('Training epoches:', training_epoches)
    print('Training time, m:', round(training_time_sec/ 60., 2))

    if task:
        ind_max = np.argmax(list(ndcg.values())) * config['skip_epochs']
        for metric_name, metric_value in metrics[ind_max].items():
            log.report_single_value(name=f'val_{metric_name}', value=round(metric_value, 4))
        log.report_single_value(name='train_peak_mem_mb', value=round(peak_training_memory_bytes/ 1024. / 1024., 2))
        log.report_single_value(name='full_train_peak_mem_mb', value=round(full_peak_training_memory_bytes/ 1024. / 1024., 2))
        log.report_single_value(name='train_epoches', value=training_epoches)
        log.report_single_value(name='train_time_m', value=round(training_time_sec/ 60., 2))

    return model


if __name__ == "__main__":

    main()
