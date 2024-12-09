from absl import app
from absl import flags
from ml_collections import config_flags

from data import get_dataset, data_to_sequences, data_to_sequences_rating
from eval_utils import sasrec_model_scoring

from time import time
from functools import reduce
import uuid
from tqdm import tqdm
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd

from data import get_dataset, data_to_sequences, SequentialDataset
from utils import topn_recommendations, downvote_seen_items
from eval_utils import model_evaluate, sasrec_model_scoring, get_test_scores

from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor
from rl_ope.fqe import RLDatasetOnline, RLDatasetOnlineVal, FQE
from rl_ope.utils import prepare_svd, extract_states_actions, extract_states_actions_val
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import logging
import wandb

def main(_):
    args = FLAGS.config

    device = args.device
    seed = args.seed
    
    os.environ["WANDB_API_KEY"] = "YOUR API KEY" # Change to your W&B profile if you need it
    os.environ["WANDB_MODE"] = "online"
    
    wandb_conf = dict(
        project= "FQE-DT4REC",
        group= "FQE-DT4REC",
        name= "FQE",
        params=dict(
            fqe_params=args.fqe_params,
            n_neg_samples=args.n_neg_samples,
            subseq_len=args.subseq_len,
            samples_per_user=args.samples_per_user
        )
    )
    
    wandb.init(
        config=wandb_conf["params"],
        project=wandb_conf["project"],
        group=wandb_conf["group"],
        name=wandb_conf["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()
    
    logger = logging.getLogger("fqe")
    fh = logging.FileHandler('logs/run_fqe.log')
    fh.setLevel(logging.INFO) # or any level you want
    logger.addHandler(fh)

    values_path = f"./saved_values/values{args.alg_type}_pen={args.optim_conf.weight_decay},"\
                  f"bs={args.fqe_params.bs},"\
                  f"nneg={args.n_neg_samples},"\
                  f"subseq_len={args.subseq_len},"\
                  f"binary_rew={args.binary_rew},"\
                  f"seed={seed}"\
                  f"rank={args.rank}.npy"

    fqe_w_path = f"./saved_fqes/fqe{args.alg_type}_pen={args.optim_conf.weight_decay},"\
                 f"bs={args.fqe_params.bs},"\
                 f"nneg={args.n_neg_samples},"\
                 f"subseq_len={args.subseq_len},"\
                 f"binary_rew={args.binary_rew},"\
                 f"seed={seed}"\
                 f"rank={args.rank}.pt"

    data_path = "/home/hdilab/amgimranov/RECE/mv1m/ml-1m.zip"


    training_temp, data_description_temp, testset_valid_temp_cut, testset_temp, holdout_valid_temp_cut, _ = get_dataset(local_file=data_path,
                                                                                         splitting='temporal_full',
                                                                                         q=0.8)

    # training_temp = pd.read_csv('../sasrec_zvuk/training_temp.csv')
    # testset_valid_temp_cut = pd.read_csv('../sasrec_zvuk/testset_valid_temp_cut.csv')
    # holdout_valid_temp_cut = pd.read_csv('../sasrec_zvuk/holdout_valid_temp_cut.csv')
    # data_description_temp = {'users': 'userid',
    #  'items': 'itemid',
    #  'order': 'timestamp',
    #  'n_users': training_temp.userid.nunique(),
    #  'n_items': training_temp.itemid.max()
    # }
    
    training_temp['rating'] = np.ones(training_temp.shape[0])
    testset_valid_temp_cut['rating'] = np.ones(testset_valid_temp_cut.shape[0])
    
    training_temp = training_temp[:2000000]

    # s = training_temp['rating']
    # if args.binary_rew:
    #     training_temp['rating'] = s.where(s >= 3, 0).mask(s >= 3, 1)
    # else:
    #     training_temp['rating'] = np.ones(s.shape[0], dtype=np.int32)

    from dt4rec.dt4rec_utils import load_model
    model_e = load_model(args.config_e.chkpt_path).to(device)

    n_actions = model_e.state_repr.item_embeddings.num_embeddings
    model_e.item_num = n_actions
    model_e.pad_token = n_actions - 1
    all_actions = np.arange(n_actions, dtype=np.int32)

    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    item_embs = None

    states, next_states, actions, ratings, sampled_seqs, actions_neg, full_sequences = extract_states_actions(training_temp,
                                                                                                 model_e, #model_D
                                                                                                 args.subseq_len,
                                                                                                 data_description_temp,
                                                                                                 device,
                                                                                                 n_neg_samples=args.n_neg_samples,
                                                                                                 samples_per_user=args.samples_per_user)
    np.save("./actions_neg.npy", actions_neg)
    actions_neg = actions_neg[:, :1]
    print("extract_states_actions_val")

    states_val, actions_val, ratings_val, _, seqs_val = extract_states_actions_val(testset_valid_temp_cut,
                                                                     model_e, #model_D
                                                                     args.subseq_len_val,
                                                                     data_description_temp,
                                                                     device)

    print("extract_states_actions_val")

    full_sequences_val = data_to_sequences(testset_valid_temp_cut, data_description_temp)

    state_dim = len(states[0])

    dataset_config = {
        "states": states,
        "next_states": next_states,
        "actions": actions,
        "rewards": ratings,
        "all_actions": all_actions,
        "sampled_seqs": sampled_seqs,
        "actions_neg": actions_neg,
        "full_sequences": full_sequences,
        "n": args.subseq_len,
        "pad_token": None,
        "samples_per_user": args.samples_per_user,
        "n_neg_samples": args.n_neg_samples,
    }

    dataset_val_config = {
        "states": states_val,
        "actions": actions_val,
        "rewards": ratings_val,
        "seqs": seqs_val,
        "full_sequences": full_sequences_val,
        "n": args.subseq_len_val,
        "pad_token": None
    }

    dataset_config["pad_token"] = model_e.pad_token
    dataset_val_config["pad_token"] = model_e.pad_token

    dataset = RLDatasetOnline(dataset_config)
    val_dataset = RLDatasetOnlineVal(dataset_val_config)
    
    fqe_config = {
        "dataset": dataset,
        "val_dataset": val_dataset,
        "pi_e": model_e,
        "item_emb": model_e.state_repr.item_embeddings.weight,
        "optim_conf": args.optim_conf,
        "n_epochs": args.fqe_params.n_epochs,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "action_dim": n_actions,
        "hidden_size": args.fqe_params.hidden_size,
        "gamma": args.gamma,
        "tau": -1,
        "n_sampled_actions": 1000,
        "use_action_emb": False,
        "device": device
    }

    fqe = FQE(**fqe_config)

    values = fqe.train(batch_size=args.fqe_params.bs, plot_info=False)

    torch.save(fqe.q.state_dict(), fqe_w_path)

    np.save(values_path, values)


if __name__ == "__main__":
    # python run_fqe.py --config=config.py:SASRec --config.config_e.chkpt_path=./saved_models/model_e0.pt --config.values_path=./saved_values/values_0.npy
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config")

    app.run(main)















