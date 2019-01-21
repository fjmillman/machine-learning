import numpy as np
from utils import *


class LinearRegression:
    def __init__(self, alpha, epochs, debug):
        self.alpha = alpha
        self.epochs = epochs
        self.cost = None
        self.theta = None
        self.debug = debug

    def gradient_descent(self, X_train, y_train):
        """
        Perform gradient descent on the training data
        """
        for i in range(self.epochs):
            loss = X_train.dot(self.theta) - y_train
            self.theta = self.theta - (self.alpha / X_train.shape[0]) * X_train.T.dot(loss)
            self.cost[i] = np.sum(loss ** 2) / (2 * X_train.shape[0])

    def train(self, X_train, y_train):
        """
        Train the model with the given training data.
        """
        X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        self.theta = np.ones((X_train.shape[1], 1))
        self.cost = np.zeros(self.epochs)
        self.gradient_descent(X_train, y_train)

        if self.debug:
            plot_data(np.arange(self.epochs), self.cost, 'Epochs', 'Cost',
                      f'Cost over the number of epochs for alpha = {self.alpha}', 'lr')

    def test(self, X_test, y_test):
        """
        Calculate the coefficient of determination.
        """
        X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        total_sum_of_squares = np.sum((y_test - np.mean(y_test)) ** 2) / (X_test.shape[0] - X_test.shape[1])
        residual_sum_of_squares = np.sum((y_test - X_test.dot(self.theta)) ** 2) / (X_test.shape[0] - 1)
        score = 1 - (residual_sum_of_squares / total_sum_of_squares)

        print(f'Alpha = {self.alpha} | Epochs = {self.epochs} | Score: {score}')

        return score

    def optimise(self, X_train, y_train, X_test, y_test):
        """
        Optimise the model for the hyperparameter alpha.
        """
        original_alpha = self.alpha
        alpha_scores = []
        alpha_bounds = np.linspace(original_alpha * 0.5, original_alpha * 1.5, 10)
        for i, alpha in enumerate(alpha_bounds):
            self.alpha = alpha
            self.train(X_train, y_train)
            score = self.test(X_test, y_test)
            alpha_scores.append((alpha, score))
        alpha_scores.sort(key=lambda score: score[1], reverse=True)
        best_alpha, best_score = alpha_scores[0]
        print(f'Best Alpha = {best_alpha} | Epochs = {self.epochs} | Score = {best_score}')

        if self.debug:
            alpha_scores.sort(key=lambda score: score[0])
            plot_data(alpha_bounds, [r_score for _, r_score in alpha_scores], 'Alpha', 'Score', 'Score over alpha', 'lr')

        self.alpha = best_alpha
