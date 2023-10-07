# Author: Yann DURIEZ
# Date: 26/09/2023
# Project: Classification Based on Probability
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    corresponding_target = []
    mean = []

    for i in range(len(features)):
        if targets[i] == selected_class:
            corresponding_target.append(features[i])
    
    corresponding_target = np.array(corresponding_target)

    for j in range(features.shape[1]):
        mean.append(np.mean(corresponding_target[:, j]))
        
    return np.array(mean)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    corresponding_target = []
    for i in range(len(features)):
        if targets[i] == selected_class:
            corresponding_target.append(features[i])
    
    corresponding_target = np.array(corresponding_target)

    return np.cov(corresponding_target, rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    
    return multivariate_normal(mean = class_mean, cov = class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))

    likelihoods = np.zeros((test_features.shape[0], len(classes)))

    for i in range(test_features.shape[0]):
        for j in range(len(classes)):
            likelihoods[i, j] = likelihood_of_class(test_features[i, :], means[j], covs[j],)

    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''

    return np.argmax(likelihoods, axis=1)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    pr_proba = []

    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))  
        pr_proba.append(np.sum(train_targets == class_label) / len(train_targets))
        
    #print(pr_proba)
    
    aposterior_likelihood = np.zeros((test_features.shape[0], len(classes)))

    for i in range(test_features.shape[0]):
        for j in range(len(classes)):
            likelihood = likelihood_of_class(test_features[i, :], means[j], covs[j],)
            aposterior_likelihood[i, j] = likelihood * pr_proba[j]

    return np.array(aposterior_likelihood)




def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    
    length_predictions = len(prediction)
    matrix = np.zeros((length_predictions, length_predictions), int)

    for i in range(len(target)):
        current_class = target[i]
        predicted_class = prediction[i]
        matrix[predicted_class][current_class] += 1
        
    return matrix


def accuracy(
        prediction: np.ndarray, 
        target: np.ndarray
        ) -> np.ndarray:
    correct_predictions = np.sum(prediction == target)    
    accuracy = 100*correct_predictions / len(prediction)
    return accuracy


if __name__ == '__main__':

    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)

    #1.1
    print(mean_of_class(train_features, train_targets, 0))
    #1.2
    print(covar_of_class(train_features, train_targets, 0))

    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)

    #1.3
    print(likelihood_of_class(test_features[0, :], class_mean, class_cov))
    
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    
    #1.4
    print(likelihoods)
    #1.5
    print(predict(likelihoods))
    
    aposteriori = maximum_aposteriori(train_features, train_targets, test_features, classes)
    
    #2.1
    print(aposteriori)
    #2.2
    print(confusion_matrix(predict(likelihoods), test_targets))

    print(f'Accuracy maximum likelihood : {accuracy(predict(likelihoods), test_targets)} %')

    print(predict(aposteriori))
    print(confusion_matrix(predict(aposteriori), test_targets))

    print(f'Accuracy maximum aposteriori: {accuracy(predict(aposteriori), test_targets)} %')
    