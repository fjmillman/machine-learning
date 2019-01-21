from Node import Node

import numpy as np


class RegressionTree:
    def __init__(self, n_features, max_depth):
        self.n_features = n_features
        self.max_depth = max_depth
        self.root = None
        self.leaves = None

    def variance(self, X):
        """
        Calculate the variance of a given split of data.
        """
        return np.sum(np.square(X - (np.sum(X) / X.shape[0]))) / X.shape[0]

    def calculate_variance(self, X_train, X_left, X_right):
        """
        Calculate the total variance between the data and its two split sets.
        """
        X_n = len(X_train)

        return (len(X_left) / X_n * self.variance(X_left)) + (len(X_right) / X_n * self.variance(X_right))

    def find_split(self, X_train):
        """
        Perform variance reduction to find the best split in the data for a select number of features
        """
        best = {'variance': np.inf}
        for i in np.random.choice(range(X_train.shape[1]), self.n_features, replace=False):
            for split in X_train[:, i]:
                if split == np.max(X_train[:, i]) or split == np.min(X_train[:, i]):
                    continue
                left_indices = np.where(X_train[:, i] < split)[0]
                right_indices = np.where(X_train[:, i] > split)[0]
                variance = self.calculate_variance(X_train[:, i], X_train[left_indices, i], X_train[right_indices, i])
                if variance < best['variance']:
                    best = {'feature': i,
                            'split': split,
                            'variance': variance,
                            'left_indices': left_indices,
                            'right_indices': right_indices}

        return best['feature'], best['split'], best['variance'], best['left_indices'], best['right_indices']

    def build(self, X_train, y_train):
        """
        Build the tree by generating a node at each level which is either designated a leaf node with the
        average of the training data as its result or a branch node with a left and a right node taking a split
        in the training data.
        """
        node = Node()
        if self.max_depth == 1 or X_train.shape[0] == 1 or y_train.shape[0] == 1:
            node.leaf = True
            node.result = np.sum(y_train) / y_train.shape[0]
        else:
            self.max_depth -= 1
            node.feature, node.split, node.infogain, left_indices, right_indices = self.find_split(X_train)
            node.left = self.build(X_train[left_indices, :], y_train[left_indices])
            node.right = self.build(X_train[right_indices, :], y_train[right_indices])

        return node

    def train(self,  X_train, y_train):
        """
        Train the model by building a tree from a root node.
        """
        self.root = self.build(X_train, y_train)

    def transverse(self, X_test, node, indices):
        """
        Transverse the tree using the given test data point to find the relevant leaf node and its prediction.
        """
        if node.leaf:
            self.leaves[indices] = node.result
        else:
            move_left = X_test[indices, node.feature] < node.split
            left_indices = indices[move_left]
            right_indices = indices[np.logical_not(move_left)]
            if left_indices.shape[0] > 0:
                self.transverse(X_test, node.left, left_indices)
            if right_indices.shape[0] > 0:
                self.transverse(X_test, node.right, right_indices)

    def predict_one(self, node, X_test):
        """
        Predict the result for a single test data point for the purpose of validation.
        """
        if node.leaf:
            return node.result
        else:
            if X_test[node.feature] < node.split:
                return self.predict_one(node.left, X_test)
            else:
                return self.predict_one(node.right, X_test)

    def predict(self, X_test):
        """
        Predict the results for each test data point.
        """
        self.leaves = np.empty(X_test.shape[0], dtype=float)
        self.leaves.fill(-1)
        indices = np.arange(X_test.shape[0])
        self.transverse(X_test, self.root, indices)

        return self.leaves

    def test(self, X_test, y_test):
        """
        Test the model with test data.
        """
        prediction = self.predict_one(self.root, X_test[0])
        print(f'Prediction | X = {X_test[0]} | y = {y_test[0]} | y prediction = {prediction}')

        self.print_tree(self.root, 2)

        predictions = self.predict(X_test)
        rmse = np.sqrt(np.sum(np.square(np.subtract(predictions, y_test))) / len(predictions))
        print(f'Max Depth = {self.max_depth} | RMSE = {rmse}')

    def print_tree(self, tree, indent=0, level=0):
        """
        Pretty print the regression tree.
        """
        indent_space = indent * level * ' '
        level = level + 1

        if tree.leaf:
            print(f"{indent_space}predict {tree.result}")
        else:
            print('')
            print(f"{indent_space}[x{tree.feature} <={tree.split}]")
            self.print_tree(tree.left, indent, level)

            print('')
            print(f"{indent_space}[x{tree.feature} >{tree.split}]")
            self.print_tree(tree.right, indent, level)
