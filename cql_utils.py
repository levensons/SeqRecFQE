from src.rl.cql_dqn import *
from data import get_dataset, data_to_sequences, SequentialDataset
from train import prepare_sasrec_model, train_sasrec_epoch, downvote_seen_items, sasrec_model_scoring, topn_recommendations, model_evaluate
from rl_ope.utils import prepare_svd
import gc
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from time import time
from clearml import Task, Logger


def prepare_cql_model(config, sasrec_model, data_description, optimizers=None):
    state_dim = data_description['n_items']+2
    action_dim = data_description['n_items']+2

    max_action = float(1)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed)


    q_1 = FullyConnectedQFunction(
        128,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers
    ).to(config.device)

    q_2 = FullyConnectedQFunction(128, action_dim, config.orthogonal_init, config.q_n_hidden_layers).to(
        config.device
    )
    q_1_optimizer = torch.optim.Adam(list(q_1.parameters()), config.qf_lr)
    q_2_optimizer = torch.optim.Adam(list(q_2.parameters()), config.qf_lr)

    kwargs = {
        "body": sasrec_model,
        "body_optimizer": optimizers,
        "q_1": q_1,
        "q_2": q_2,
        "q_1_optimizer": q_1_optimizer,
        "q_2_optimizer": q_2_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": 1,
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
        "cql_negative_samples": 10
    }

    trainer = DQNCQL(**kwargs)

    trainer.num_embeddings = sasrec_model.pos_emb.num_embeddings

    return trainer