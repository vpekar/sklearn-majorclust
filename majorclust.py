
import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.metrics.pairwise import linear_kernel

from sklearn.metrics.pairwise import cosine_similarity


def get_sim_matrix(X, threshold=0.9):
    """Pairwise cosine similarities in X. Cancel out similarities smaller
    than the threshold, set the diagonal values to 0.
    """
    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, 0.0)
    sim_matrix[sim_matrix < threshold] = 0.0
    return sim_matrix


def majorclust(sim_matrix):
    """Actual MajorClust algorithm
    """
    t = False
    indices = np.arange(sim_matrix.shape[0])
    while not t:
        t = True
        for index in np.arange(sim_matrix.shape[0]):
            # check if all the sims of the word are not zeros
            weights = sim_matrix[index]
            if weights[weights > 0.0].shape[0] == 0:
                continue
            # aggregating edge weights
            new_index = np.argmax(np.bincount(indices, weights=weights))
            if indices[new_index] != indices[index]:
                indices[index] = indices[new_index]
                t = False
    return indices


class MajorClust(BaseEstimator, ClusterMixin):

    def __init__(self, sim_threshold=0.99):
        self.sim_threshold = sim_threshold

    def fit(self, X, y=None):
        """Parameters
        ----------
        X : array or csr_matrix of shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse='csr')
        sim_matrix = get_sim_matrix(X, self.sim_threshold)
        self.labels_ = majorclust(sim_matrix)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
