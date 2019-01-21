import io
import csv
import zipfile
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def read_dataset(filename):
    """
    Read the data from the given filename to form a dataset array
    along with a list of the unique labels that appear in the dataset.
    """
    with zipfile.ZipFile(f'{filename}.zip') as zf:
        with zf.open(f'{filename}.csv') as f:
            reader = csv.reader(io.TextIOWrapper(f))
            next(reader)
            return np.array([[float(value) for value in row] for row in reader], dtype=np.float32)


def get_toy_dataset(number):
    """
    Create a toy dataset with no noise for validation and optimisation.
    """
    X = np.linspace(-50, 50, 101).reshape(-1, 1)
    y = np.sin(X) if number is 0 else 3 * X + 2
    return np.c_[X, y]


def split_dataset(dataset, percentage=0.6, random=True, is_print=True):
    """
    Split the given dataset into a training and testing dataset with
    the split given by the percentage.
    """
    np.random.seed(0)
    dataset = dataset[np.random.permutation(dataset.shape[0])]
    split = int(dataset.shape[0] * percentage)
    train, test = dataset[:split, :], dataset[split:, :]
    if not random:
        train, test = np.sort(train, axis=0), np.sort(test, axis=0)
    X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
    y_train, y_test = y_train.reshape((y_train.shape[0], 1)), y_test.reshape((y_test.shape[0], 1))
    if is_print:
        print(f'Training set size: {X_train.shape[0]}')
        print(f'Test set size: {X_test.shape[0]}')
    return X_train, y_train, X_test, y_test


def normalise_data(data):
    """
    Normalise the data
    """
    return (data - np.mean(data, dtype=np.float64)) / np.std(data)


def plot_data(X, y, X_label, y_label, title, model):
    """
    Plot the given data with labels, a title, and the model using it.
    """
    plt.plot(X, y)
    plt.xlabel(X_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'plots/{model}-{title.replace(" ", "-").lower()}.png')
    plt.close()


def plot_features(data, labels):
    """
    Plot each input feature against the output for visualisation.
    """
    colors = cm.rainbow(np.linspace(0, 1, data.shape[1]))
    for i in range(data.shape[1]):
        plt.figure(figsize=(10, 10))
        plt.plot(data[:, i], labels, marker='.', linestyle='none', color=colors[i])
        plt.title(f'Relationship between feature {i + 1} and 22')
        plt.tight_layout()
        plt.savefig(f'plots/feature-comparison-{i + 1}-22.png')
        plt.close()


def plot_gp(mu, std, X_test, samples, sigma_n=None, X_train=None, y_train=None, title=None):
    """
    Plot a Gaussian Process
    """
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(X_test, sample, alpha=0.1, color='black')
    plt.plot(X_test, mu, 'r')
    if X_train is not None:
        plt.errorbar(X_train, y_train, yerr=sigma_n, color='k', fmt='o')
        plt.fill_between(x=X_test.squeeze(), y1=mu.squeeze() - 1.96 * std, y2=mu.squeeze() + 1.96 * std, color='gray', alpha=0.25)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('$f(x)$')
    plt.savefig(f'plots/gp-{title.replace(" ", "-").lower()}.png')
    plt.close()
