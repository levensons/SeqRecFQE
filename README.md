# SeqRecFQE

## Code structure

- `dt4rec/`: folder with files for model training(`train.py`). Also, it contains model instance(`gpt1.py`) which is used in FQE;
- `rl_ope/`: folder with code for the FQE preprocessing(`utils.py`) and training(`fqe.py`);
- `src/`: folder with the instance of CQL model(`cql_dqn.py`) and replay buffer(`rec_replay_buffer.py`);
- `ssknn`: folder with its own realization of FQE(`rl_ope/`) and contains jupyter notebook for the model training and evaluation(`ml1_sknn.ipynb`);
- `config.py`: file with the configuration for various evaluated algorithms;
- `data.py`, `eval_utils.py`, `metrics.py`, `model.py`, `sampler.py`, `preprocessing.py`, `train.py`, `utils.py`: files to train and evaluate SasRec model;
- `cql_utils.py`: file which contains code to create CQL+SasRec model;
- `train_cql.ipynb`, `train_sasrec.ipynb`: notebooks which run training for CQL+SasRec and SasRec models;
- `run_fqe.py`, `run_fqe_cql.py`, `run_fqe_dt4rec.py`: script files which run FQE for the corresponding model.


## Train models
- To train SasRec model you can follow the `train_sasrec.ipynb` file;
- To train CQL model you can follow the `train_cql.ipynb` file;
- To train DT4REC model(for example, MovieLens-1m) you should use the following command
```bash
python train.py <name_of_experiment> movielens -tbs=64
```
- To train SKNN model you can follow the `sknn/ml1_sknn.ipynb` file.

**Important**: if you run code on MovieLens-1m then you should use `downvote_seen_item` function on validation. In the case of Zvuk dataset you shouldn't because within one user tracks can be repeated.

## Run FQE
To prepare this code you should write paths in `config.py` for the model(evaluated) weights.

- To run FQE algorithm for the SasRec models you should use the following command
```bash
python run_fqe.py --config=config.py:SASRec0
```
In this case you can switch between different models just changing `SASRec0` on `SASRec1` or `SASRec2`.
- To run FQE algorithm for the CQL with SasRec backbone you should use the following command
```bash
python run_fqe_cql.py --config=config.py:CQL
```
- To run FQE for the DT4REC model you should use
```bash
python run_fqe_dt4rec.py --config=config.py:DT4REC
```
- To run FQE algorithm for the SKNN model you can follow the notebook `sknn/ml1_sknn.ipynb`.

In these scripts you should change the data loading paths. Also, if you run code on a Zvuk dataset with large action space, you can change the parameter `use_action_emb` and set the parameter `action_dim` in according to the dimension of the item embeddings.