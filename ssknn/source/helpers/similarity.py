import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(type, m1, m2):
    if type == 'jaccard':
        similarity = jaccard_similarity(m1, m2)
    elif type == 'weighted_jaccard':
        similarity = weighted_jaccard_similarity(m1, m2)
    elif type == 'cosine':
        similarity = cosine_similarity(m1, m2, dense_output=False)
    else:
        raise ValueError(f'Unknown similarity type: {type}')
    return similarity

def jaccard_similarity(A, B):
    '''
    Computes the jaccard similarity index between the rows of two input matrices.
    The matrices are binarized.
    Jaccard(u, v) = \frac{\sum_{i=1}^k \min(u_k, v_k)}{\sum_{i=1}^k \max(u_k, v_k)}
    
    Args:
        A (scipy.sparse.csr_matrix): n_users_A x n_items
        B (scipy.sparse.csr_matrix): n_users_B x n_items

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix of shape (n_users_A, n_users_B) containing the similarities between users
    '''
    assert A.shape[1] == B.shape[1]
    A_bin = A.astype('bool').astype('int')
    B_bin = B.astype('bool').astype('int')

    numerator = A_bin @ B_bin.T
    denominator = A_bin.sum(axis=1) + B_bin.sum(axis=1).T - A_bin @ B_bin.T
    similarity = csr_matrix(numerator / denominator)
    return similarity


def weighted_jaccard_index(u, v):
    '''
    Computes weighted jaccard index between the rows of matrices u and v.
    
    '''
    numerator = u.minimum(v).sum(axis=1).A.squeeze()
    denominator = u.maximum(v).sum(axis=1).A.squeeze()
    return (numerator / denominator)


def weighted_jaccard_similarity(A, B):
    '''
    Computes the weighted jaccard similarity index between the rows of two input matrices.
    Weighted_jaccard(u, v) = \frac{\sum_{i=1}^k \min(u_k, v_k)}{\sum_{i=1}^k \max(u_k, v_k)}
    
    Args:
        A (scipy.sparse.csr_matrix): n_users_A x n_items
        B (scipy.sparse.csr_matrix): n_users_B x n_items

    Returns:
        scipy.sparse.csr_matrix: A sparse matrix of shape (n_users_A, n_users_B) containing the similarities between users
    '''
    assert A.shape[1] == B.shape[1]
    similarity = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        # construct a new matrix A_tile
        # of the same shape as B,
        # each row of matrix A_tile
        # is equal to the i-th row of matrix A
        row = csr_matrix(A[i, :])
        rows, cols = B.shape
        
        A_tile = csr_matrix((np.tile(row.data, rows), np.tile(row.indices, rows),
                                np.arange(0, rows*row.nnz + 1, row.nnz)), shape=B.shape)
        # compute the similarity between user i and users in matrix B
        similarity[i, :] = weighted_jaccard_index(A_tile, B)
    return csr_matrix(similarity)