import numpy as np
from utils import *


class NearestNeighbour:
    def __init__(self, k, debug):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.debug = debug

    def fit(self, X_train, y_train):
        """
        Fit the training data to the model.
        """
        self.X_train, self.y_train, = X_train, y_train

    def euclidean_distance(self, feature):
        """
        Calculate the Euclidean Distance between a given feature and the fitted data.
        """
        return np.sqrt(np.sum((self.X_train - feature) ** 2, axis=1))

    def predict(self, X_test):
        """
        Calculate the predictions for the testing data.
        """
        predictions = []
        for feature in X_test:
            distances = [(i, distance) for i, distance in enumerate(self.euclidean_distance(feature))]
            distances.sort(key=lambda distance: distance[1])
            k_distances = distances[:self.k]
            prediction = np.sum(self.y_train[[i for i, distance in k_distances]]) / self.k
            predictions.append(prediction)
        return predictions

    def test(self, X_test, y_test):
        """
        Test the fitted model with testing data and the k hyperparameter.
        """
        predictions = self.predict(X_test)
        rmse = np.sqrt(np.sum(np.subtract(predictions, y_test) ** 2) / len(predictions))
        print(f'K = {self.k} | RMSE = {rmse}')

        return rmse

    def optimise(self, X_train, y_train, X_test, y_test):
        """
        Optimise the model for the hyperparameter k.
        """
        self.fit(X_train, y_train)

        k_scores = []
        max_k = self.k + 1
        for k in range(1, max_k):
            self.k = k
            rmse = self.test(X_test, y_test)
            k_scores.append((k, rmse))
        k_scores.sort(key=lambda score: score[1])
        best_k, best_rmse = k_scores[0]
        print(f'Best K = {best_k} | RMSE = {best_rmse}')

        if self.debug:
            k_scores.sort(key=lambda score: score[0])
            plot_data([k for k, _ in k_scores], [rmse for _, rmse in k_scores], 'K', 'RMSE', 'RMSE over the number of K points', 'nn')

        self.k = best_k
