# Adapted from https://github.com/rn5l/session-rec/blob/master/algorithms/sbr_adapter/factorization/fossil.py

import numpy as np
from tqdm.notebook import tqdm
from source.dataprep.dataprep import generate_histories

class FOSSIL:
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        # generator for interaction sampling
        self.rng = np.random.default_rng(seed=model_config['seed'])
        
        self.sigma = model_config.get('sigma', 0.01)
        self.n_users = model_config['n_users']
        self.n_items = model_config['n_items']
        self.alpha = model_config['alpha']
        self.lr = model_config['lr']
        self.L = model_config['markov_order']
        self.reg = model_config['regularization']
        self.dim = model_config['dim']
        
        # random initialization of model's factors
        self.P = self.rng.normal(loc=0.0, scale=self.sigma, size=(model_config['n_items'], model_config['dim']))
        self.Q = self.rng.normal(loc=0.0, scale=self.sigma, size=(model_config['n_items'], model_config['dim']))
        self.eta = self.rng.normal(loc=0.0, scale=self.sigma, size=(model_config['n_users'], model_config['markov_order']))
        self.eta_bias = np.zeros(self.L)
        self.bias = np.zeros(self.n_items)
        
        self.n_iters = model_config.get('epoch_iterations', 100000)

    def sigmoid(self, x, cutoff = 10.0):
        x_cutoff = max(min(cutoff, x), -cutoff)
        return 1.0 / (1.0 + np.exp(-x_cutoff))

    def compute_score(self, user_id, curr_hist, item):
        # if the user's history is shorter than the 
        # order of markov chain, use the effective length of sequence
        length = min(self.L, len(curr_hist))
        
        long_term = (np.power(len(curr_hist), -self.alpha) * self.P[curr_hist, :].sum(axis=0))
        short_term = np.dot((self.eta_bias + self.eta[user_id, :])[:length], self.P[curr_hist[:-length-1:-1], :])

        return self.bias[item] + np.dot(long_term + short_term, self.Q[item, :])
        
    def fit_partial(self, interactions, data_description, n_epochs, n_iters=None):
        # fit the model. repeated calls to this method will
        # cause training to resume from the current model state
        
        histories = generate_histories(interactions, data_description)
        iterations = n_iters if n_iters is not None else self.n_iters
        
        for epoch in range(n_epochs):
            for _ in tqdm(range(iterations), desc='Training'):
                
                user = self.rng.integers(0, self.n_users)
                user_history = histories[user]
                pos_index = self.rng.integers(1, len(histories[user]))
                curr_hist = user_history[:pos_index + 1]
                neg = self.rng.choice(np.setdiff1d(np.arange(self.model_config['n_items']), user_history))

                pos = curr_hist[-1]
                curr_hist = curr_hist[:-1]
                length = min(self.L, len(curr_hist))

                long_term = np.power(len(curr_hist), -self.alpha) * self.P[curr_hist, :].sum(axis=0)
                short_term = np.dot((self.eta_bias + self.eta[user, :])[:length], self.P[curr_hist[:-length-1:-1], :])

                x_pos = self.compute_score(user, curr_hist, pos)
                x_neg = self.compute_score(user, curr_hist, neg)

                delta = 1.0 - self.sigmoid(x_pos - x_neg)
                
                # Gradients
                V_upd = self.lr * ( delta * np.power(len(curr_hist), -self.alpha) * (self.Q[pos, :] - self.Q[neg, :]) - self.reg * self.P[curr_hist, :])
                V_upd2 = self.lr * delta *  np.outer((self.eta_bias + self.eta[user, :])[:length], self.Q[pos, :] - self.Q[neg, :])
                Q_pos_upd = self.lr * ( delta * (long_term + short_term) - self.reg * self.Q[pos, :])
                Q_neg_upd = self.lr * ( -delta * (long_term + short_term) - self.reg * self.Q[neg, :])
                bias_pos_upd = self.lr * (delta - self.reg * self.bias[pos])
                bias_neg_upd = self.lr * (- delta - self.reg * self.bias[neg])
                eta_bias_upd = self.lr * (delta * np.dot(self.P[curr_hist[:-length-1:-1], :], self.Q[pos, :] - self.Q[neg, :]) - self.reg * self.eta_bias[:length])
                eta_upd = self.lr * (delta * np.dot(self.P[curr_hist[:-length-1:-1], :], self.Q[pos, :] - self.Q[neg, :]) - self.reg * self.eta[user, :length])

                # Update
                self.P[curr_hist, :] += V_upd
                self.P[curr_hist[:-length-1:-1], :] += V_upd2
                self.Q[pos, :] += Q_pos_upd
                self.Q[neg, :] += Q_neg_upd
                self.bias[pos] += bias_pos_upd
                self.bias[neg] += bias_neg_upd
                self.eta_bias[:length] += eta_bias_upd
                self.eta[user, :length] += eta_upd
                
                        
    def folding_in(self, interactions, data_description, n_epochs=1, n_iters = None):
        # this function allows to do folding-in of new users.
        # it does the "half-step" of gd to update user embeddings
        # without updating the item factors in MF and MC parts
        iterations = n_iters if n_iters is not None else self.n_iters
        
        histories = generate_histories(interactions, data_description)
        
        eta_warmstart = self.rng.normal(loc=0.0, scale=self.sigma, size=(len(histories), self.L))
        
        for epoch in range(n_epochs):
            for _ in tqdm(range(iterations), desc='Folding-in'):
                
                user = self.rng.integers(0, len(histories))
                user_history = histories[user]
                pos_index = self.rng.integers(1, len(histories[user]))
                curr_hist = user_history[:pos_index + 1]
                neg = self.rng.choice(np.setdiff1d(np.arange(self.model_config['n_items']), user_history))

                pos = curr_hist[-1]
                curr_hist = curr_hist[:-1]
                length = min(self.L, len(curr_hist))
  
                # Compute error
                x_pos = self.compute_score(user, curr_hist, pos)
                x_neg = self.compute_score(user, curr_hist, neg)
                delta = self.sigmoid(x_pos - x_neg) # sigmoid of the error
                
                # Compute Update
                eta_upd = self.lr * (delta * np.dot(self.P[curr_hist[:-length-1:-1], :], self.Q[pos, :] - self.Q[neg, :]) - self.reg * self.eta[user, :length])
                eta_warmstart[user, :length] += eta_upd

        # generate scores for warm users
        scores = np.zeros((len(histories), self.n_items))
        
        for user in tqdm(range(len(histories)), desc='Scoring'):
            curr_hist = histories[user]
            length = min(self.L, len(curr_hist))
            
            long_term = np.power(len(curr_hist), -self.alpha) * self.P[curr_hist, :].sum(axis=0)
            short_term = np.dot((self.eta_bias + self.eta[user, :])[:length], self.P[curr_hist[:-length-1:-1], :])
            
            scores[user, :] = self.bias + np.dot(long_term + short_term, self.Q.T)

        return scores

        