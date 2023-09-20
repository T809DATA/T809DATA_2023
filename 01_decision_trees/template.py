# Author: Yann DURIEZ
# Date: 28/08/2023
# Project: Decision Trees
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''

    proba = [0] * len(classes)

    for n in range(len(targets)):
        proba[targets[n]] += 1

    for m in range(len(proba)) :
        proba[m] /= len(targets)
    
    return proba


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    
    split = features[:, split_feature_index]

    set1 = split < theta
    set2 = split >= theta

    features_1 = features[set1]
    targets_1 = targets[set1]

    features_2 = features[set2]
    targets_2 = targets[set2]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    my_sum=0
    for i in classes:
        proba_class = np.count_nonzero(targets == i) / len(targets)
        my_sum += np.power(proba_class, 2)

    impurity= 0.5 * (1 - my_sum)

    return impurity


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
    ) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]

    return (t1.shape[0]*g1 / n) + (t2.shape[0]*g2 / n)


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
    ) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    
    return weighted_impurity(t_1, t_2, classes)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
    ) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        min_value = features[:,i].min()
        max_value = features[:,i].max()
        thetas = np.linspace(min_value, max_value, num_tries+2)[1:-1]

        # iterate thresholds
        for theta in thetas:
            gini = total_gini_impurity(features, targets, classes, i, theta)

            if gini < best_gini:
                best_gini = gini
                best_theta = theta
                best_dim = i

    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
        ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        prediction = self.tree.predict(self.test_features)

        right_predictions = np.sum(prediction == self.test_targets)
        nb_samples = len(self.test_targets)
        accuracy = right_predictions / nb_samples
        
        return accuracy

    def plot(self):
        plot_tree(self.tree, filled=True)
        plt.show()

    def guess(self):
        
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        nb_classes = len(self.classes)
        prediction = self.tree.predict(self.test_features)
        matrix = np.zeros((nb_classes, nb_classes), int)

        for i in range(len(self.test_targets)):
            current_class = self.test_targets[i]
            predicted_class = prediction[i]
            matrix[predicted_class][current_class] += 1
        
        return matrix


if __name__ == '__main__':

    features, targets, classes = load_iris()
    
    #1.1
    print (prior([0, 0, 1], [0, 1]), prior([0, 2, 3, 3], [0, 1, 2, 3]))
    #1.2
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
    print ('F1 : ', f_1)
    print ('F2 : ', f_2)
    #1.3
    print(gini_impurity(t_1, classes))
    print(gini_impurity(t_2, classes))
    #1.4
    print (weighted_impurity(t_1, t_2, classes))
    #1.5
    print(total_gini_impurity(features, targets, classes, 2, 4.65))
    #1.6
    print(brute_best_split(features, targets, classes, 30))

    #2.1
    dt = IrisTreeTrainer(features, targets, classes=classes)
    dt.train()
    #2.2
    print(f'The accuracy is: {dt.accuracy()}')
    #2.3
    dt.plot()
    #2.4
    print(f'I guessed: {dt.guess()}')
    #2.5
    print(f'The true targets are: {dt.test_targets}')
    print(dt.confusion_matrix())