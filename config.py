from ml_collections import config_dict

def get_config(alg_type):
    device = "cuda:0"
    
    config_sasrec = config_dict.ConfigDict(
            {
                "gen_model": "prepare_sasrec_model",
                "params": config_dict.ConfigDict(
                    {
                        "manual_seed": 123,
                        "sampler_seed": 123,
                        "init_emb_svd": None,
                        "lin_layer_dim": -1,
                        "num_epochs": 5, #3 10 22 100&dropout0.9&hd32&bs1000
                        "maxlen": 100,
                        "hidden_units": 128,
                        "dropout_rate": 0.3,
                        "num_blocks": 2,
                        "num_heads": 2,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "fwd_type": 'bce',
                        "l2_emb": 0,
                        "patience": 10,
                        "skip_epochs": 1,
                        "n_neg_samples": 600,
                        "sampling": 'without_rep'
                    }
                ),
                "chkpt_path": "./saved_models/model_e0_nonsvd.pt"
            }
        )
    

    config_e = {
        "DT4REC": config_dict.ConfigDict(
            {
                "gen_model": "load_model",
                "chkpt_path": "./models/first_zvuk.pt"
            }
        ),
        
        "SASRec0": config_dict.ConfigDict(
            {
                "gen_model": "prepare_sasrec_model",
                "params": config_dict.ConfigDict(
                    {
                        "manual_seed": 123,
                        "sampler_seed": 123,
                        "init_emb_svd": None,
                        "lin_layer_dim": -1,
                        "num_epochs": 5, #3 10 22 100&dropout0.9&hd32&bs1000
                        "maxlen": 100,
                        "hidden_units": 128,
                        "dropout_rate": 0.3,
                        "num_blocks": 2,
                        "num_heads": 2,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "fwd_type": 'bce',
                        "l2_emb": 0,
                        "patience": 10,
                        "skip_epochs": 1,
                        "n_neg_samples": 600,
                        "sampling": 'without_rep'
                    }
                ),
                "chkpt_path": "./saved_models/model_e0_nonsvd.pt"
            }
        ),

        "SASRec1": config_dict.ConfigDict(
            {
                "gen_model": "prepare_sasrec_model",
                "params": config_dict.ConfigDict(
                    {
                        "manual_seed": 123,
                        "sampler_seed": 123,
                        "init_emb_svd": None,
                        "lin_layer_dim": -1,
                        "num_epochs": 2, #3 10 22 100&dropout0.9&hd32&bs1000
                        "maxlen": 100,
                        "hidden_units": 128,
                        "dropout_rate": 0.3,
                        "num_blocks": 2,
                        "num_heads": 2,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "fwd_type": 'bce',
                        "l2_emb": 0,
                        "patience": 10,
                        "skip_epochs": 1,
                        "n_neg_samples": 600,
                        "sampling": 'without_rep'
                    }
                ),
                "chkpt_path": "./saved_models/model_e1_nonsvd.pt"
            }
        ),

        "SASRec2": config_dict.ConfigDict(
            {
                "gen_model": "prepare_sasrec_model",
                "params": config_dict.ConfigDict(
                    {
                        "manual_seed": 123,
                        "sampler_seed": 123,
                        "init_emb_svd": None,
                        "lin_layer_dim": -1,
                        "num_epochs": 5, #3 10 22 100&dropout0.9&hd32&bs1000
                        "maxlen": 100,
                        "hidden_units": 128,
                        "dropout_rate": 0.9,
                        "num_blocks": 2,
                        "num_heads": 2,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "fwd_type": 'bce',
                        "l2_emb": 0,
                        "patience": 10,
                        "skip_epochs": 1,
                        "n_neg_samples": 600,
                        "sampling": 'without_rep'
                    }
                ),
                "chkpt_path": "./saved_models/model_e2_nonsvd.pt"
            }
        ),

        "CQL": config_dict.ConfigDict(
            {
                "gen_model": "prepare_cql_model",
                "params": config_dict.ConfigDict(
                    {
                        "orthogonal_init": True,
                        "q_n_hidden_layers": 1,
                        "qf_lr": 3e-4,
                        "batch_size": 64,
                        "device": device,
                        "bc_steps": 100000,
                        "cql_alpha": 100.0,
                        "env": "MovieLens",
                        "project": "CQL-SASREC",
                        "group": "CQL-SASREC",
                        "name": "CQL"
                        #cql_negative_samples = 10
                    }
                ),
                "chkpt_path": "./saved_models/sasrec_cql.pt"
            }
        )
    }[alg_type]

    config_D = config_dict.ConfigDict(
        {
            "gen_model": "prepare_sasrec_model",
            "params": config_dict.ConfigDict(
                {
                    "manual_seed": 123,
                    "sampler_seed": 123,
                    "init_emb_svd": None,
                    "lin_layer_dim": -1,
                    "num_epochs": 100,
                    "maxlen": 100,
                    "hidden_units": 64,
                    "dropout_rate": 0.3,
                    "num_blocks": 2,
                    "num_heads": 1,
                    "batch_size": 128,
                    "learning_rate": 1e-3,
                    "fwd_type": 'ce',
                    "l2_emb": 0,
                    "patience": 10,
                    "skip_epochs": 1,
                    "n_neg_samples": 0,
                    "sampling": "no_sampling"
                }
            ),
            "chkpt_path": "./saved_models/model_D_sasrec.pt"
        }
    )

    config = config_dict.ConfigDict(
        {
            "config_sasrec": config_sasrec,
            "config_e": config_e,
            "config_D": config_D
        }
    )

    config.optim_conf = config_dict.ConfigDict(
        {
            "lr": 3e-4,
            "weight_decay": 1e-4
        }
    )

    config.fqe_params = config_dict.ConfigDict(
        {
            "n_epochs": 150,
            "hidden_size": 512,
            "bs": 512
        }
    )

    config.subseq_len = 5
    config.subseq_len_val = 5
    config.n_neg_samples = 100
    config.gamma = 0.98
    config.device = device
    config.rank = -1
    config.seed = 42
    config.binary_rew = False
    config.samples_per_user = 100

    config.alg_type = alg_type

    return config






