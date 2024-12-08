import numpy as np

class Random:
    def __init__(self, model_config={}) -> None:
        # for reproducibility, not a hyperparameter
        self.seed = model_config.get('seed', 2024)
        self.rng = np.random.default_rng(seed=self.seed)

    def build(self, data, data_description):
        self.n_items = data_description['n_items']

    def recommend(self, data, data_description):
        n_users = data.nunique()[data_description['users']]
        n_items = data_description['n_items']
        return self.rng.random((n_users, n_items))
    

class Popular:
    def __init__(self, model_config=None) -> None:
        pass

    def build(self, data, data_description):
        item_popularity = data[data_description['items']].value_counts()
        n_items = item_popularity.index.max() + 1
        popularity_scores = np.zeros(n_items,)
        popularity_scores[item_popularity.index] = item_popularity.values
        self.popularity_scores = popularity_scores

    def recommend(self, data, data_description):
        n_users = data.nunique()[data_description['users']]
        return np.tile(self.popularity_scores, (n_users, 1))