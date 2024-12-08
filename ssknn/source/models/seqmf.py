import numpy as np
from source.dataprep.dataprep import generate_histories, generate_interactions_matrix
from scipy.sparse import diags, coo_matrix
from tqdm.notebook import tqdm

class SeqMF:
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.last_n = model_config['last_n']
        self.gamma = model_config['gamma']
        self.beta = model_config['beta']
        self.lambd = model_config['lambda']
        self.n_items = model_config['n_items']
        self.dim = model_config['dim']
        sigma = model_config.get('sigma', 0.01)
        self.P = sigma * np.random.randn(model_config['n_users'], model_config['dim'])
        self.Q = sigma * np.random.randn(model_config['n_items'], model_config['dim'])
        self.S = []
        self.C = []
        
    
    def h(self, user, history):
        last_n_items = history[-self.last_n:]
        return self.Q @ self.Q[last_n_items, :].sum(axis=0)
    
    def precompute(self, data, data_description):
        '''
        Builds C_u and S_u matrices for each user.
        C_u is weighted frequency of user's item interaction
        S_u is the item transition matrix for the user
        '''
        
        histories = generate_histories(data, data_description)
        interactions = generate_interactions_matrix(data, data_description)
        
        for u in range(len(histories)):
            weights = interactions[u, :].A.squeeze() ** self.gamma
            C_u = diags(weights / weights.sum())
            self.C.append(C_u)

            rules = {}
            denominator = np.zeros((self.n_items))
            for i in range(1, len(histories[u])):
                if (histories[u][i - 1], histories[u][i]) not in rules:
                    rules[(histories[u][i - 1], histories[u][i])] = 0
                rules[(histories[u][i - 1], histories[u][i])] += 1
                denominator[histories[u][i - 1]] += 1
                
            # create a sparse matrix
            items, values = zip(*rules.items())
            i1, i2 = zip(*items)
            matrix_shape = (data_description['n_items'], data_description['n_items'])
            S_u = coo_matrix((values, (list(i1), list(i2))), shape=matrix_shape).tocsr()
                
            S_u = S_u @ diags(np.divide(1.0, denominator, where=(denominator!=0), out=denominator))

            self.S.append(S_u)

    def fit(self, data, data_description, n_epochs):
        # !!! hybrid optimization scheme !!!
        # user factors are updated using ALS
        # item factors are updated using GD

        if len(self.S) == 0:
            self.precompute(data, data_description)
            
        interactions = generate_interactions_matrix(data, data_description)
        histories = generate_histories(data, data_description)
                
        for epoch in range(n_epochs):
            # updating user factors through ALS scheme
            # for u in tqdm(range(len(histories)), desc='update user'):
            for u in tqdm(range(len(histories)), desc='Updating user factors'):
                self.P[u, :] = np.linalg.inv(
                    self.Q.T @ self.C[u] @ self.Q + self.lambd * np.eye(self.dim)
                    ) @ self.Q.T @ self.C[u] @ (interactions[u, :].toarray().squeeze() - self.h(u, histories[u]))
                
            # updating item factors via GD
            grad = self.lambd * self.Q
            u_ids = np.random.permutation(len(histories))
            for u in tqdm(u_ids, desc='Updating item factors'):
                D_u = diags(
                    self.C[u] @ (self.Q @ self.P[u, :] + self.h(u, histories[u]) - interactions[u, :].toarray().squeeze())
                    )
                grad += (np.outer(D_u.sum(axis=1), self.P[u, :]) + (D_u @ self.S[u] + self.S[u].T @ D_u) @ self.Q)
            self.Q -= self.beta * grad
    
    def folding_in(self, data, data_description):
        C_warm = []
        warm_interactions = generate_interactions_matrix(data, data_description, rebase_users=True)
        warm_histories = generate_histories(data, data_description)
        n_warm_users = warm_interactions.shape[0]
        
        # initialize warm user embeddings P_warm and matrices C_warm
        for u in range(n_warm_users):
            weights = warm_interactions[u, :].A.squeeze() ** self.gamma
            C_u = diags(weights / weights.sum())
            C_warm.append(C_u)
        P_warm = np.zeros((n_warm_users, self.dim))
        
        # do the half-step of training procedure, that is the ALS step to get 
        # the warm user embeddings
        for u in range(n_warm_users):
            P_warm[u, :] = np.linalg.inv(
                self.Q.T @ C_warm[u] @ self.Q + self.lambd * np.eye(self.dim)
                ) @ self.Q.T @ C_warm[u] @ (warm_interactions[u, :].toarray().squeeze() - self.h(u, warm_histories[u]))

        # generate scores for warm users
        scores = np.zeros((n_warm_users, self.n_items))
        for u in range(n_warm_users):
            scores[u, :] = self.Q @ P_warm[u, :] + self.h(u, warm_histories[u])
        return scores
