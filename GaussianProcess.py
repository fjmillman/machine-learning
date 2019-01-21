from utils import *
import numpy as np


class GaussianProcess:
    def __init__(self, l, sigma_f, sigma_n, debug):
        self.l = l
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.debug = debug

    def fit(self, X_train, y_train):
        """
        Fit the training data to the model and calculate their covariance matrix and its inverse.
        """
        self.X_train = X_train
        self.y_train = y_train
        K = self.covariance_matrix(self.X_train, self.X_train)
        self.K_inv = np.linalg.inv(K + self.sigma_n ** 2 * np.eye(K.shape[0]))

    def squared_exponential(self, X_p, X_q):
        """
        Computes the squared exponential covariance function using the hyperparameters sigma_f for signal variance,
        l for length-scale, and sigma_n for noise variance. The equality denotes the Kronecker delta function which
        disregards the noise variance if the two variables are equal.
        """
        return self.sigma_f ** 2 * np.exp(-1 / (2 * self.l ** 2) * (X_p - X_q) ** 2) + self.sigma_n ** 2 * (X_p == X_q)

    def covariance_matrix(self, X_p, X_q):
        """
        Compute the covariance matrix by performing the covariance kernel function element-wise on each pair of values.
        """
        K = np.zeros((X_p.shape[0], X_q.shape[0]))
        for i, p in enumerate(X_p):
            for j, q in enumerate(X_q):
                K[i][j] = self.squared_exponential(p, q)
        return K

    def posterior_samples(self, X_test, size=1):
        """
        Obtain samples for the posterior mean and covariance.
        """
        K_x = self.covariance_matrix(self.X_train, X_test)
        K_xx = self.covariance_matrix(X_test, X_test)

        mu = K_x.T.dot(self.K_inv).dot(self.y_train)
        cov = K_xx - K_x.T.dot(self.K_inv).dot(K_x)
        std = np.sqrt(np.diag(cov))
        posteriors = np.random.multivariate_normal(mu.squeeze(), cov, size)

        return posteriors, mu, cov, std

    def prior_samples(self, X_test, size=1):
        """
        Obtain samples for the prior.
        """
        mu = np.zeros(X_test.shape[0])
        cov = self.covariance_matrix(X_test, X_test)
        std = np.sqrt(np.diag(cov))
        priors = np.random.multivariate_normal(mu.squeeze(), cov, size)

        return priors, mu, cov, std

    def test(self):
        """
        Plot predictions for the data given.
        """
        X_prior = np.linspace(-50, 50, 101)
        priors, mu, cov, std = self.prior_samples(X_prior, 5)
        plot_gp(mu, std, X_prior, priors, title='Prior Observations')

        posteriors, mu, cov, std = self.posterior_samples(X_prior, 50)
        plot_gp(mu, std, X_prior, posteriors, X_train=self.X_train, y_train=self.y_train, title='Posterior Observations')
