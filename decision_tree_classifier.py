import numpy as np

# --- 1. Node Class ---
class Node:
    """A single node in the Decision Tree."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # For internal node (split node)
        self.feature_index = feature_index # Index of the feature to split on
        self.threshold = threshold         # Value of the feature to split on
        self.left = left                   # Left child node (True condition)
        self.right = right                 # Right child node (False condition)
        
        # For leaf node (terminal node)
        self.value = value                 # The predicted class label

# --- 2. Decision Tree Classifier Class ---
class DecisionTreeClassifier:
    """Decision Tree Classifier implementation using Gini Impurity."""
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Builds the decision tree."""
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)

    def predict(self, X):
        """Predicts class labels for new data."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # --- Tree Construction Method ---
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping Criteria (Base Cases)
        # 1. Max depth reached
        # 2. Minimum samples for splitting not met
        # 3. All samples belong to the same class (Pure node)
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_labels == 1):
            
            # Create a leaf node with the majority class
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_split = self._best_split(X, y, n_samples, n_features)

        # If no gain in impurity, stop and create leaf node
        if best_split["gain"] == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Recursive splitting
        left_child = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
        right_child = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)

        # Create the internal node
        return Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_child,
            right=right_child
        )

    # --- Splitting/Impurity Methods ---
    def _best_split(self, X, y, n_samples, n_features):
        best_gain = -1
        best_index = -1
        best_threshold = -1
        
        # Iterate over all features
        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column) # Check splits only at unique feature values
            
            # Iterate over all possible thresholds for the current feature
            for threshold in thresholds:
                # Split the data
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                
                # Check if split resulted in non-empty nodes
                if len(y_left) > 0 and len(y_right) > 0:
                    # Calculate Gini Gain
                    current_gain = self._gini_gain(y, y_left, y_right)
                    
                    if current_gain > best_gain:
                        best_gain = current_gain
                        best_index = feature_index
                        best_threshold = threshold
                        best_X_left, best_y_left = X_left, y_left
                        best_X_right, best_y_right = X_right, y_right

        # Return the best split information
        return {
            "feature_index": best_index,
            "threshold": best_threshold,
            "X_left": best_X_left,
            "y_left": best_y_left,
            "X_right": best_X_right,
            "y_right": best_y_right,
            "gain": best_gain
        }

    def _split(self, X, y, feature_index, threshold):
        """Splits the dataset based on a feature and a threshold."""
        # True condition: X_column <= threshold goes to the left node
        left_mask = X[:, feature_index] <= threshold
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask] # Complementary mask for the right node
        
        return X_left, y_left, X_right, y_right

    def _gini_impurity(self, y):
        """Calculates the Gini Impurity of a set of labels."""
        if len(y) == 0:
            return 0
        
        # Calculate probability of each class
        p_k = np.array([np.sum(y == c) / len(y) for c in np.unique(y)])
        
        # Gini Impurity: G = 1 - sum(p_k^2)
        gini = 1.0 - np.sum(p_k**2)
        return gini

    def _gini_gain(self, parent_y, left_y, right_y):
        """Calculates the Information Gain using Gini Impurity."""
        n_parent = len(parent_y)
        n_left, n_right = len(left_y), len(right_y)

        # Weighted Gini Impurity of children
        weight_left = n_left / n_parent
        weight_right = n_right / n_parent
        
        gini_children = (weight_left * self._gini_impurity(left_y) + 
                         weight_right * self._gini_impurity(right_y))

        # Gini Gain: Gain = Gini(Parent) - Gini(Children)
        gini_gain = self._gini_impurity(parent_y) - gini_children
        return gini_gain

    def _most_common_label(self, y):
        """Returns the majority class label."""
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    # --- Prediction Method ---
    def _traverse_tree(self, x, node):
        """Traverses the tree to find the leaf node prediction."""
        # Check if we have reached a leaf node
        if node.value is not None:
            return node.value

        # Check the feature condition
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)