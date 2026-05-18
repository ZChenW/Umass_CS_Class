import numpy as np

# import pandas as pd
from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_trees=10,
        max_lenght=None,
        min_samples_split=2,
        min_gain=1e-6,
        node_chose_limit=None,
        random_seed=None,
    ):
        self.n_trees = n_trees
        self.max_lenght = max_lenght
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.node_chose_limit = node_chose_limit
        self.random_seed = np.random.RandomState(random_seed)
        self.trees = []

    def fit(self, df):
        self.trees = []
        for _ in range(self.n_trees):
            indices = self.random_seed.choice(len(df), size=len(df), replace=True)
            data = df.iloc[indices].reset_index(drop=True)

            tree = DecisionTree(
                max_lenght=self.max_lenght,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain,
                node_chose_limit=self.node_chose_limit,
                random_seed=self.random_seed.randint(0, 10000),
            )
            tree.fit(data)
            self.trees.append(tree)
        return self

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predit(X))  # predictions.shape (n_trees, n_samples)
        predictions = np.array(predictions).T  # ppredictions.shape (n_samples, n_trees)

        finial_predictions = []
        for i in predictions:
            value, count = np.unique(i, return_counts=True)
            finial_predictions.append(value[np.argmax(count)])
        return np.array(finial_predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))

