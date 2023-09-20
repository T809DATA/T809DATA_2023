# Author: Yann DURIEZ
# Date: 12/09/2023
# Project: 03-Sequential estimation
# Acknowledgements: 
#

from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
    ) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    covariance = np.identity(k)*var*var
    x_array = (np.random.multivariate_normal(mean, covariance, size = n))
    
    return x_array


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
    ) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + ((x - mu) / n)


def _plot_sequence_estimate():
    np.random.seed(1234)

    n = 100
    k = 3
    var = 4
    init_estimates = np.array([0, 0, 0])
    data = gen_data(n, k, init_estimates, var)
    
    mean = init_estimates
    estimates = []

    for i in range(data.shape[0]):
        mean = update_sequence_mean(init_estimates, data[i], i+1)
        estimates.append(mean)
    
    estimates = np.array(estimates)

    plt.figure()
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    return np.square(y - y_hat)


def _plot_mean_square_error():
    n = 100
    k = 3
    var = 4
    init_estimates = np.array([0, 0, 0])
    data = gen_data(n, k, init_estimates, var)
    
    mean = init_estimates
    estimates = []
    data_mean = np.mean(data)
    error = []

    for i in range(data.shape[0]):
        mean = update_sequence_mean(init_estimates, data[i], i+1)
        estimates.append(mean)
        sq_error = _square_error(data_mean, mean)
        error.append(np.mean(sq_error))
    

    error = np.array(error)

    plt.figure()
    plt.plot(range(n), error, label='Error')
    plt.legend(loc='upper center')
    plt.show()


if __name__ == '__main__':
    
    #1.1
    np.random.seed(1234)
    print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    np.random.seed(1234)
    print(gen_data(5, 1, np.array([0.5]), 0.5))
    
    #1.2
    np.random.seed(1234)
    X = gen_data(300, 3, np.array([0, 1, -1]), 1.73)
    scatter_3d_data(X)
    bar_per_axis(X)
    
    #1.4
    np.random.seed(1234)
    mean = np.mean(X, 0)
    new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    print(update_sequence_mean(mean, new_x, X.shape[0]+1))
    
    #1.5
    np.random.seed(1234)
    _plot_sequence_estimate()
    
    #1.6
    np.random.seed(1234)
    _plot_mean_square_error()