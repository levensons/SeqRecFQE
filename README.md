# SeqRecFQE

## Train models
- To train SasRec model you can follow the `train_sasrec.ipynb` file;
- To train CQL model you can follow the `train_cql.ipynb` file;
- To train DT4REC model(for example, MovieLens-1m) you should use the following command
```bash
python train.py <name_of_experiment> movielens -tbs=64
```
- To train SKNN model you can follow the `sknn/ml1_sknn.ipynb` file.

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