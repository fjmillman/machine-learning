import click
from utils import *

from NearestNeighbour import NearestNeighbour
from LinearRegression import LinearRegression
from RegressionForest import RegressionForest
from GaussianProcess import GaussianProcess


Models = ['NN', 'LR', 'RF', 'GP']


@click.command()
@click.option('--model', '-m', prompt='Model', help='The model you wish to fit data to')
@click.option('--dataset', '-d', prompt='Data', help='The dataset you wish to fit to the model')
@click.option('--optimise', '-o', is_flag=True, help='Optimise the selected model', type=bool, default=False)
@click.option('--split', '-s', help='The split in the dataset for training and testing data', type=float, default=0.3)
@click.option('--k', '-k', help='The value of k', type=int, default=5)
@click.option('--alpha', '-a', help='The learning rate alpha', type=float, default=0.01)
@click.option('--epochs', '-e', help='The number of epochs to run', type=int, default=1000)
@click.option('--trees', '-t', help='The number of trees to generate', type=int, default=1)
@click.option('--max_depth', '-md', help='Max depth of decision trees', type=int, default=1)
@click.option('--length', '-l', help='The length of the kernel', type=float, default=1.0)
@click.option('--sigma_f', '-sf', help='The magnitude of the kernel', type=float, default=10.0)
@click.option('--sigma_n', '-sn', help='The noise of the kernel', type=float, default=10.0)
@click.option('--features', '-f', help='The number of features to train', type=int, default=1)
@click.option('--debug', '-db', is_flag=True, help='Plot the data for debugging', type=bool, default=False)
def main(model, dataset, optimise, split, k, alpha, epochs, trees, max_depth, length, sigma_f, sigma_n, features, debug):
    if model not in Models:
        print('Please specify one of the following models:')
        print(' - NN for Nearest Neighbour')
        print(' - LR for Linear Regression')
        print(' - RF for Regression Forest')
        print(' - GP for Gaussian Process')

    if dataset == 'toy_dataset':
            dataset = get_toy_dataset(0 if model in ['NN', 'GP'] else 1)
    else:
        dataset = read_dataset(dataset)

    if model == 'NN':
        nn = NearestNeighbour(k=k, debug=debug)
        dataset = normalise_data(dataset)
        X_train, y_train, X_test, y_test = split_dataset(dataset, percentage=split)
        if optimise:
            nn.optimise(X_train, y_train, X_test, y_test)
        else:
            nn.fit(X_train, y_train)
            nn.test(X_test, y_test)

    if model == 'LR':
        lr = LinearRegression(alpha=alpha, epochs=epochs, debug=debug)
        dataset = normalise_data(dataset)
        X_train, y_train, X_test, y_test = split_dataset(dataset, percentage=split)
        if optimise:
            lr.optimise(X_train, y_train, X_test, y_test)
        else:
            lr.train(X_train, y_train)
            lr.test(X_test, y_test)

    if model == 'RF':
        rf = RegressionForest(n_features=features, n_estimators=trees, max_depth=max_depth, split=split, debug=debug)
        dataset = normalise_data(dataset)
        X_train, y_train, X_test, y_test = split_dataset(dataset, percentage=split)
        if optimise:
            rf.optimise(X_train, y_train, X_test, y_test)
        else:
            rf.train(X_train, y_train)
            rf.test(X_test, y_test)

    if model == 'GP':
        gp = GaussianProcess(l=length, sigma_f=sigma_f, sigma_n=sigma_n, debug=debug)
        X_train, y_train, _, _ = split_dataset(dataset, random=False, percentage=0.8)
        gp.fit(X_train, y_train)
        gp.test()


if __name__ == '__main__':
    main()
