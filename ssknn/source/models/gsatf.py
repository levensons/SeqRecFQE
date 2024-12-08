from source.dataprep.dataprep import generate_tensor
import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

def unfold(idx, vals, shape, mode):
    # idx, vals, shape - the description of sparse tensor
    # mode - the mode for unfolding
    
    unfolding_shape = 1
    contraction_modes = []
    for i, dim in enumerate(shape):
        if i != mode:
            unfolding_shape *= dim
            contraction_modes.append(i)
    matrix_shape = (shape[mode], unfolding_shape)

    J = []
    for i in range(len(shape)):
        if i != mode:
            J_i = 1
            for j in range(i + 1, len(shape)):
                if j != mode:
                    J_i *= shape[j]
            J.append(J_i)

    J = np.array(J)
    x = idx[mode, :]
    y = (idx[contraction_modes, :] * J[:, np.newaxis]).sum(axis=0)

    matrix = csr_matrix((vals, (x, y)), shape=matrix_shape)
    return matrix


def random_orthonormal(d1, d2, rng):
    assert d1 >= d2, 'The matrix is not tall, d1 < d2'
    
    # builds a random matrix of size (d1, d2) with orthonormal columns
    A = rng.random((d1, d2))
    Q, _ = np.linalg.qr(A)
    return Q

from scipy.sparse.linalg import svds, LinearOperator
from scipy.sparse import csr_matrix, eye, diags
from scipy.linalg import solve_triangular


class PositionalTF:
    def __init__(self, config) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed=config['seed'])
        
        self.n_iter = config['n_iter']
        
        self.U = random_orthonormal(config['n_users'], config['dim_u'], self.rng) # user factor
        self.V = random_orthonormal(config['n_items'], config['dim_i'], self.rng) # item factor
        self.W = random_orthonormal(config['seq_len'], config['dim_k'], self.rng) # pos factor
        
        # attention matrix
        self.A = np.zeros((config['seq_len'], config['seq_len'])) 
        for i in range(self.A.shape[0]):
            for j in range(i + 1):
                self.A[i][j] = (i - j + 1) ** -self.config['attn_deg']
        
        # precompute for convenience
        self.AW = self.A @ self.W
        
    def matvec_seq(self, q):
        Q = q.reshape((self.U.shape[1], self.V.shape[1]))
        vector = (self.U @ Q @ self.V.T).reshape(-1)
        return self.A.T.dot(self.unfoldings[2].dot(vector))
    def rmatvec_seq(self, q):
        Q = (self.unfoldings[2].T.dot(self.A @ q)).reshape(
            (self.U.shape[0], self.V.shape[0])
            )
        return (self.U.T @ Q @ self.V).reshape(-1)

    def matvec_user(self, q):
        Q = q.reshape((self.V.shape[1], self.W.shape[1]))
        vector = (self.V @ Q @ self.AW.T).reshape(-1)
        
        return self.unfoldings[0].dot(vector)
    def rmatvec_user(self, q):
        Q = (self.unfoldings[0].T.dot(q)).reshape((self.V.shape[0], self.A.shape[0]))
        return (self.V.T @ Q @ self.AW).reshape(-1)

    def matvec_item(self, q):
        Q = q.reshape((self.U.shape[1], self.W.shape[1]))
        vector = (self.U @ Q @ self.AW.T).reshape(-1)
        return self.unfoldings[1].dot(vector)
    def rmatvec_item(self, q):
        Q = (self.unfoldings[1].T.dot(q)).reshape((self.U.shape[0], self.A.shape[0]))
        return (self.U.T @ Q @ self.AW).reshape(-1)

    def unfolding(self, idx, vals, shape):
        self.unfoldings = []
        for mode in range(len(shape)):
            self.unfoldings.append(unfold(idx, vals, shape, mode))

    def hooi(self, n_iter, verbose=False):
        if verbose:
            iterator = tqdm
        else:
            iterator = lambda x: x
        core_norms = []
        for iter in iterator(range(n_iter)):
            linop_u = LinearOperator(
                shape=(self.config['n_users'], self.config['dim_i'] * self.config['dim_k']),
                matvec=self.matvec_user,
                rmatvec=self.rmatvec_user
                )
            factor_u, sigma_u, _ = svds(linop_u, self.config['dim_u'], return_singular_vectors="u")
            self.U = factor_u[:, np.argsort(-sigma_u)]

            linop_v = LinearOperator(
                shape=(self.config['n_items'], self.config['dim_u'] * self.config['dim_k']),
                matvec=self.matvec_item,
                rmatvec=self.rmatvec_item
                )
            factor_v, sigma_v, _ = svds(linop_v, self.config['dim_i'], return_singular_vectors="u")
            self.V = factor_v[:, np.argsort(-sigma_v)]

            linop_w = LinearOperator(
                shape=(
                    self.config['seq_len'],
                    self.config['dim_u'] * self.config['dim_i']
                    ),
                matvec=self.matvec_seq,
                rmatvec=self.rmatvec_seq
                )
            factor_w, sigma_w, _ = svds(linop_w,
                                        self.config['dim_k'],
                                        return_singular_vectors="u")
            self.W = factor_w[:, np.argsort(-sigma_w)]
            
            # update AW with new W
            self.AW = self.A @ self.W
            
            # save core norms for future analysis
            core_norms.append(np.linalg.norm(sigma_u))
            core_norms.append(np.linalg.norm(sigma_v))
            core_norms.append(np.linalg.norm(sigma_w))

        return core_norms

    def build(self, data, data_description, verbose=False):
        n_users = data_description["n_users"]
        n_items = data_description["n_items"]
        max_pos = data_description["n_pos"]
        shape = (n_users, n_items, max_pos)
        
        # here we construct the tensor 
        # precompute unfoldings along each mode
        # perform HOOI algorithm
    
        idx, val = generate_tensor(data, data_description)
        self.unfolding(idx, val, shape)
        core_norms = self.hooi(self.n_iter, verbose=verbose)
        self.core_norms = core_norms

    def recommend(self, warm_data, data_description):
        # P - user's history of interactions, n_items x n_pos
        # toprecs(P, n) = argmax(VV^T P SAW w'_K)
        # A^TW' = W
        
        # since A is triangular, 
        # the system A^T W_hat = W 
        # can be solved efficiently
        W_hat = solve_triangular(self.A.T, self.W)

        # shift operator for sequences
        S = diags(np.ones(data_description['n_pos'] - 1), -1)

        projector = S.dot(
            self.AW.dot(
                W_hat[-1, :]
                )
            )

        # create positional tensor from warm data
        idx_warm, val_warm = generate_tensor(warm_data, data_description, rebase_users=True)

        preferences = np.zeros(
            (warm_data.nunique()[data_description['users']],
             data_description['n_items'])
            )
        
        for i in range(idx_warm.shape[1]):
            user, item, pos = idx_warm[0][i], idx_warm[1][i], idx_warm[2][i]
            preferences[user, item] += projector[pos]

        # we calculated the preferences
        # containing sequential information,
        # now the regular folding-in
        scores = (preferences @ self.V) @ self.V.T

        return scores