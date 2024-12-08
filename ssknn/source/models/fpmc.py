import numpy as np
from source.dataprep.dataprep import generate_histories
from tqdm.notebook import tqdm

class FPMC:
    def __init__(self, model_config) -> None:
        self.model_config = model_config 
        # generator for interaction sampling
        self.rng = np.random.default_rng(seed=model_config['seed'])
        self.n_iters = model_config.get('epoch_iterations', 100000)
        self.n_users = model_config['n_users']
        self.n_items = model_config['n_items']
        self.dim = model_config['dim']
        # random initialization of model's factors
        self.sigma = model_config.get('sigma', 0.01)
        self.V_LI = self.rng.normal(loc=0.0, scale=self.sigma, size=(model_config['n_items'], model_config['dim']))
        self.V_IL = self.rng.normal(loc=0.0, scale=self.sigma, size=(model_config['n_items'], model_config['dim']))
        self.V_IU = self.rng.normal(loc=0.0, scale=self.sigma, size=(model_config['n_items'], model_config['dim']))
        self.V_UI = self.rng.normal(loc=0.0, scale=self.sigma, size=(model_config['n_users'], model_config['dim']))
        
    
    def sigmoid(self, x, cutoff = 10.0):
        x_cutoff = max(min(cutoff, x), -cutoff)
        return 1.0 / (1.0 + np.exp(-x_cutoff))

        
    def fit_partial(self, interactions, data_description, n_epochs=1, n_iters=None):
        # fit the model. repeated calls to this method will
        # cause training to resume from the current model state
        histories = generate_histories(interactions, data_description)
        
        # to speed up the model training, 
        iterations = n_iters if n_iters else len(interactions)
        for epoch in range(n_epochs):
            for _ in tqdm(range(iterations), desc='Training'):
                # sample the triplet user, positive, negative
                user = self.rng.integers(0, self.n_users)
                pos_index = self.rng.integers(0, len(histories[user]))
                pos = histories[user][pos_index]
                neg = self.rng.choice(
                    np.setdiff1d(
                        np.arange(self.model_config['n_items']),
                        histories[user]
                        )
                    )
                
                if pos_index == 0:
                    # the first item in interaction history has no predecessor,
                    # so we should calculate only MF part
                    x_upos = self.V_UI[user, :] @ self.V_IU[pos, :]
                    x_uneg = self.V_UI[user, :] @ self.V_IU[neg, :]
                else:
                    prev = histories[user][pos_index - 1]
                    x_upos = (self.V_UI[user, :] @ self.V_IU[pos, :]
                              + self.V_IL[pos, :] @ self.V_LI[prev, :])
                    x_uneg = (self.V_UI[user, :] @ self.V_IU[neg, :]
                              + self.V_IL[neg, :] @ self.V_LI[prev, :])
                delta = 1.0 - self.sigmoid(x_upos - x_uneg)
                
                # update MF part
                self.V_UI[user, :] += self.model_config['alpha'] * (
                    delta * (self.V_IU[pos, :] - self.V_IU[neg, :])
                    - self.model_config['L_UI'] * self.V_UI[user, :]
                    )
                self.V_IU[pos, :] += self.model_config['alpha'] * (
                    delta * self.V_UI[user, :] - self.model_config['L_IU'] * self.V_IU[pos, :]
                    )
                self.V_IU[neg, :] += self.model_config['alpha'] * (
                    -delta * self.V_UI[user, :] - self.model_config['L_IU'] * self.V_IU[neg, :]
                    )
                
                if pos_index > 0:
                    # if user's interaction is not the first interaction,
                    # update MC part
                    prev = histories[user][pos_index - 1]
                    self.V_IL[pos, :] += self.model_config['alpha'] * (
                        delta * self.V_LI[prev, :] - self.model_config['L_IL'] * self.V_IL[pos, :]
                        )
                    self.V_IL[neg, :] += self.model_config['alpha'] * (
                        -delta * self.V_LI[prev, :] - self.model_config['L_IL'] * self.V_IL[neg, :]
                        )
                    self.V_LI[prev, :] += self.model_config['alpha'] * (
                        delta * (self.V_IL[pos, :] - self.V_IL[neg, :])
                        - self.model_config['L_LI'] * self.V_LI[prev, :]
                        )
                        
    def folding_in(self, interactions, data_description, n_epochs=1, n_iters=None):
        # this function allows to do folding-in of new users.
        # it does the "half-step" of gd to update user embeddings
        # without updating the item factors in MF and MC parts
        histories = generate_histories(interactions, data_description)
        
        # initialize the warm user embeddings
        V_UI_warmstart = self.rng.normal(loc=0.0, scale=self.sigma, size=(len(histories), self.dim))
        
        # last interaction of each user to generate scores
        last_interactions = np.array(
            [histories[u][-1] for u in range(len(histories))]
            )
        
        iterations = n_iters if n_iters else self.n_iters
        
        for epoch in range(n_epochs):
            for _ in tqdm(range(iterations), desc='Folding-in'):
                # sample the triplet user, positive, negative
                user = self.rng.integers(0, len(histories))
                pos_index = self.rng.integers(0, len(histories[user]))
                pos = histories[user][pos_index]
                neg = self.rng.choice(
                    np.setdiff1d(
                np.arange(self.model_config['n_items']), histories[user]))

                if pos_index == 0:
                    x_upos = V_UI_warmstart[user, :] @ self.V_IU[pos, :]
                    x_uneg = V_UI_warmstart[user, :] @ self.V_IU[neg, :]
                else:
                    prev = histories[user][pos_index - 1]
                    x_upos = V_UI_warmstart[user, :] @ self.V_IU[pos, :] + self.V_IL[pos, :] @ self.V_LI[prev, :]
                    x_uneg = V_UI_warmstart[user, :] @ self.V_IU[neg, :] + self.V_IL[neg, :] @ self.V_LI[prev, :]
                
                delta = 1.0 - self.sigmoid(x_upos - x_uneg)
                
                # update only MF user embeddings
                V_UI_warmstart[user, :] += self.model_config['alpha'] * (delta * (self.V_IU[pos, :] - self.V_IU[neg, :]) - self.model_config['L_UI'] * V_UI_warmstart[user, :])
        
        # calculate the scores using the obtained warm user embeddings
        # first term corresponds to MF part, 
        # second term models the transition from the last item 
        # in user history to another one
        scores = V_UI_warmstart[:, :] @ self.V_IU.T + self.V_LI[last_interactions, :] @ self.V_IL.T
        return scores