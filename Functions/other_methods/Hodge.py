import scipy
import numpy as np


def rankGrad(rank_vector):
    return np.subtract.outer(rank_vector, rank_vector).T


def geo_rank(M):
    def rankErrorGrad(rank_vector):
        return 2 * (
            -sum(M - rankGrad(rank_vector)) + sum((M - rankGrad(rank_vector)).T)
        )

    def rankError(rank_vector):
        return np.sum((M - rankGrad(rank_vector)) ** 2)

    best_x_vec = scipy.optimize.fmin_bfgs(
        f=rankError,
        x0=np.zeros(shape=(M.shape[0])),
        fprime=rankErrorGrad,
        maxiter=50,
        disp=False,
    )
    return best_x_vec
