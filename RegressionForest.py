from utils import split_dataset
from RegressionTree import RegressionTree

import numpy as np


class RegressionForest:
    def __init__(self, n_estimators, n_features, max_depth, split, debug):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.max_depth = max_depth
        self.split = split
        self.random_forest = []
        self.debug = debug

    def train(self, X_train, y_train):
        """
        Train a Regression Forest using the given training data by training a number of Regression Trees
        each with a random sample of the training data.
        """
        train_dataset = np.c_[(X_train, y_train)]
        for i in range(self.n_estimators):
            bootstrap_sample = train_dataset[np.random.choice(
                train_dataset.shape[0], size=int(round(train_dataset.shape[0] * self.split)), replace=True)]
            X_train, y_train, _, _ = split_dataset(bootstrap_sample, self.split, is_print=False)
            tree = RegressionTree(self.n_features, self.max_depth)
            tree.train(X_train, y_train)
            self.random_forest.append(tree)

    def test(self, X_test, y_test):
        """
        Test the model using the given test data by taking the average of the predictions obtained from
        each Regression Tree.
        """
        tree_predictions = []
        for tree in self.random_forest:
            tree_predictions.append(tree.predict(X_test))
        forest_predictions = np.sum(tree_predictions, axis=0) / self.n_estimators
        rmse = np.sqrt(np.sum(np.square(np.subtract(forest_predictions, y_test))) / len(forest_predictions))
        print(f'Number of Trees = {self.n_estimators} | Max Depth = {self.max_depth} | RMSE = {rmse}')
        return rmse

    def optimise(self, X_train, y_train, X_test, y_test):
        """
        Optimise the number of estimators hyperparameter
        """
        n_estimator_scores = []
        for n_estimators in range(1, self.n_estimators + 1):
            self.n_estimators = n_estimators
            self.train(X_train, y_train)
            rmse = self.test(X_test, y_test)
            n_estimator_scores.append((n_estimators, rmse))
        n_estimator_scores.sort(key=lambda score: score[1])
        best_n_estimators, best_rmse = n_estimator_scores[0]
        print(f'Best Number of Trees = {best_n_estimators} | Max Depth = {self.max_depth} | RMSE = {best_rmse}')

        self.n_estimators = best_n_estimators
