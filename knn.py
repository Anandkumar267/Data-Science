import numpy as np
class KNN:
    def __init__(self, k =5):
        self.k= k
    def fit(self, X, y):
        self.X = X
        self.y = y
    def distance(self, x1, x2):
        return ((x1-x2)**2).sum()**0.5
    def predict_one(self, x):
        #Computing distances from one x to all X in training set
        distances = [self.distance(x, x_train) for x_train in self.X]

        #Getting the indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        #Extracting the labels of the k nearest neighbors
        k_nearest_labels = [self.y[i] for i in k_indices]
        #Majority vote, most common class label
        most_common = np.bincount(k_nearest_labels).argmax()

        return most_common
    def predict(self, X):
        y_pred = [self.predict_one(x) for x in X]
        return np.array(y_pred)
            