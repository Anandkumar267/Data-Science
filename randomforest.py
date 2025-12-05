import numpy as np
from collections import Counter
from decision_tree_classifier import DecisionTreeClassifier
class RandomForest:
    def __init__(self, n_estimators=100, min_splits=2, max_depth=5):
        self.n_estimators= n_estimators
        self.max_depth = max_depth
        self.min_splits = min_splits
        self.trees = []
    def bootstrap_sample(self, x, y):
        n_samples = x.shape[0]
        n = np.random.choice(n_samples,n_samples, replace=1)
        return x[n], y[n]
    def fit(self, x, y):
        for _  in range(self.n_estimators):

            tree = DecisionTreeClassifier(min_split=self.min_splits, max_depth=self.max_depth)
            tree.fit(self.bootstrap_sample(x, y))
            self.trees.append(tree)
    def majority_vote(self, pred):
        return Counter(pred).most_common(1)[0][0]
    def predict(self, x):
        tree_pred = [tree.predict(x) for tree in self.trees]
        tree_pred = np.swapaxes(tree_pred, 0, 1)
        return np.array([self.majority_vote(pred) for pred in tree_pred])
