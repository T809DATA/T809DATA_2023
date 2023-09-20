# Author: DURIEZ Yann
# Date: 13/09/2023
# Project: Linear Models for Regression
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
    ) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I 
    covariance matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis 
    function output fi for each data vector x in features
    '''

    N, D = features.shape
    M = mu.shape[0]
    fi = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            covar = np.eye(D) * sigma
            fi[i, j] = multivariate_normal.pdf(features[i, :], mean=mu[j, :], cov = covar)

    return fi


def _plot_mvn(features: np.ndarray,
    mu: np.ndarray,
    sigma: float
    ) -> np.ndarray:
    
    fi = mvn_basis(features, mu, sigma)

    plt.figure(figsize=(10,4))
    
    for m in range(fi.shape[1]):
        plt.plot(fi[:, m])
    plt.show()


def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
    ) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    fiT = fi.transpose()
    dim_M = np.identity(fi.shape[1])
    w = np.linalg.inv((fiT.dot(fi)) + lamda*dim_M).dot(fiT).dot(targets)

    return w

def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    fi = mvn_basis(features, mu, sigma)
    return fi.dot(w)

#1.5

def mean_square_error(y, y_hat):
    return np.square(y - y_hat)

def update_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
    ) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + ((x - mu) / n)

def _plot_mean_square_error(features, mu, sigma, w):
    init_estimates = np.array([0, 0, 0])
    prediction = linear_model(features, mu, sigma, w)
    
    mean = init_estimates
    estimates = []
    prediction_mean = np.mean(prediction)
    error = []

    for i in range(prediction.shape[0]):
        mean = update_mean(mean, prediction[i], i+1)
        estimates.append(mean)
        sq_error = mean_square_error(mean, prediction_mean)
        error.append(np.mean(sq_error))
    

    error = np.array(error)

    plt.figure()
    plt.plot(error, label='Average Squared Error')
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    X, t = load_regression_iris()
    N, D = X.shape

    #1.1
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    print(fi)

    #1.2
    _plot_mvn(X, mu, sigma)

    #1.3
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    print(wml)

    #1.4
    prediction = linear_model(X, mu, sigma, wml)
    print(prediction)

    #1.5
    _plot_mean_square_error(X, mu, sigma, wml)