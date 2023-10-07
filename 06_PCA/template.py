# Author: Yann DURIEZ
# Date: 27/09/2023
# Project: 06_PCA
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    
    standard_x = (X - np.mean(X)) / np.std(X) 

    return standard_x


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    Xi = X[:, i]
    Xj = X[:, j]

    Xi_standardized = standardize(Xi)
    Xj_standardized = standardize(Xj)

    plt.scatter(Xi_standardized, Xj_standardized, s=5)
    

def _scatter_cancer():
    X, y = load_cancer()

    plt.figure(figsize=[12, 8])
    
    for i in range(30):
        plt.subplot(5, 6, i+1)
        scatter_standardized_dims(X, 0, i)

    plt.tight_layout()
    plt.show()


def _plot_pca_components():
    
    X, y = load_cancer()

    pca = PCA(30)
    X = standardize(X)
    pca.fit_transform(X)
    cmp = pca.components_

    plt.figure(figsize=[12, 8])

    for i in range(30):
        plt.subplot(5, 6, i+1)
        plt.plot(cmp[:, i])
        plt.title(f'PCA {i+1}')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def _plot_eigen_values():
    
    X, y = load_cancer()

    pca = PCA(30)
    X = standardize(X)
    pca.fit_transform(X)
    pca_var = pca.explained_variance_

    plt.plot(pca_var)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


def _plot_log_eigen_values():
    
    X, y = load_cancer()

    pca = PCA(30)
    X = standardize(X)
    pca.fit_transform(X)
    pca_var = pca.explained_variance_
    log = np.log10(pca_var)

    plt.plot(log)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    
    X, y = load_cancer()

    pca = PCA(30)
    X = standardize(X)
    pca.fit_transform(X)
    pca_var = pca.explained_variance_
    cum_sum = np.cumsum(pca_var) / np.sum(pca_var)

    plt.plot(cum_sum)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()


'''if __name__ == '__main__':
    #1.1
    print(standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]])))
    
    #1.2
    X = np.array([
    [1, 2, 3, 4],
    [0, 0, 0, 0],
    [4, 5, 5, 4],
    [2, 2, 2, 2],
    [8, 6, 4, 2]])
    scatter_standardized_dims(X, 0, 2)
    plt.show()

    #1.3
    _scatter_cancer()
    plt.show
    
    #2.1
    _plot_pca_components()
    plt.show

    #3.1
    _plot_eigen_values()
    plt.show

    #3.2
    _plot_log_eigen_values()
    plt.show

    #3.3
    _plot_cum_variance()
    plt.show
    '''