# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    distance = np.sqrt(np.sum(np.square(x - y)))
    return distance


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])

    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    sorted = np.argsort(distances)

    return sorted[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    classes_cnt = np.bincount(targets)
    most_common = np.argmax(classes_cnt)

    return most_common


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
    ) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest_indices = k_nearest(x, points, k)
    nearest_target = point_targets[nearest_indices]

    return vote(nearest_target, classes)

#2
import help as help

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
    ) -> np.ndarray:
    
    prediction = []

    for i in range(len(points)):
        predict = knn(points[i], help.remove_one(points, i), help.remove_one(point_targets, i), classes, k)
        prediction.append(predict)

    return np.array(prediction)
    

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    
    prediction = knn_predict(points, point_targets, classes, k)

    return np.mean(prediction == point_targets)


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    
    nb_classes = len(classes)
    prediction = knn_predict(points, point_targets, classes, k)
    matrix = np.zeros((nb_classes, nb_classes), int)

    for i in range(len(point_targets)):
        current_class = point_targets[i]
        predicted_class = prediction[i]
        matrix[predicted_class][current_class] += 1
        
    return matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    ) -> int:
    
    best_knn = []

    for i in range(1, len(points)-1):
        knn_i=knn_accuracy(points, point_targets, classes, i)
        best_knn.append(knn_i)
        
    return np.argmax(best_knn) + 1

def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
    ):
    
    colors = ['yellow', 'purple', 'blue']
    edge_colors = ['green', 'red']

    for i in range(len(points)):
        x = points[i]
        y = point_targets[i]
        prediction = knn(x, points, point_targets, classes, k)
        if prediction == y:
            edge = edge_colors[0]
        else:
            edge = edge_colors[1]
        [x, y] = points[i,:2]
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors= edge, linewidths=2)

    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()


if __name__ == '__main__':

    d, t, classes = load_iris()
    
    plot_points(d, t)
    x, points = d[0,:], d[1:, :]
    x_target, point_targets = t[0], t[1:]

    #1.1
    print(euclidian_distance(x, points[0]))
    print(euclidian_distance(x, points[50]))
    #1.2
    print(euclidian_distances(x, points))
    #1.3
    print(k_nearest(x, points, 1))
    print(k_nearest(x, points, 3))
    #1.4
    print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
    print(vote(np.array([1,1,1,1]), np.array([0,1])))
    #1.5
    print(knn(x, points, point_targets, classes, 1))
    print(knn(x, points, point_targets, classes, 5))
    print(knn(x, points, point_targets, classes, 150))
    
    #2.1
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)

    print(knn_predict(d_test, t_test, classes, 10))
    print(knn_predict(d_test, t_test, classes, 5))
    #2.2
    print(knn_accuracy(d_test, t_test, classes, 10))
    print(knn_accuracy(d_test, t_test, classes, 5))
    #2.3
    print(knn_confusion_matrix(d_test, t_test, classes, 10))
    print(knn_confusion_matrix(d_test, t_test, classes, 20))
    #2.4
    print(best_k(d_train, t_train, classes))
    #2.5
    knn_plot_points(d, t, classes, 3)