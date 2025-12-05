import numpy as np

class CustomSVM:
    """
    A Linear Support Vector Machine (SVM) implementation using 
    Stochastic Gradient Descent (SGD) and Hinge Loss.
    """
    def __init__(self, learning_rate=0.001, C=0.01, n_epochs=1000):
        # Hyperparameters
        self.learning_rate = learning_rate # Alpha in the update rule
        self.C = C                       # Regularization parameter (Cost)
        self.n_epochs = n_epochs         # Number of training iterations
        
        # Model parameters (initialized in fit)
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Trains the SVM using SGD.
        
        X: Training features (n_samples, n_features)
        y: Target labels (n_samples,). Must be in {-1, 1}.
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize weights (w) and bias (b)
        self.w = np.zeros(n_features)
        self.b = 0

        # Ensure labels are {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # 2. Start SGD training
        for epoch in range(self.n_epochs):
            # Iterate over each sample
            for idx, x_i in enumerate(X):
                # Calculate the decision function value: y_i * (w.x_i + b)
                # If this value is >= 1, the point is classified correctly and outside the margin.
                
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Case 1: Classified Correctly (Loss is 0)
                    # The gradient is just from the regularization term (1/2 * ||w||^2)
                    self.w = self.w - self.learning_rate * (2 * self.w)
                    # Bias (b) remains unchanged
                else:
                    # Case 2: Misclassified or within the margin (Loss > 0)
                    # Gradient includes regularization AND hinge loss
                    
                    # Gradient of L w.r.t w: 2*w - C * y_i * x_i
                    self.w = self.w - self.learning_rate * (2 * self.w - self.C * y_[idx] * x_i)
                    
                    # Gradient of L w.r.t b: -C * y_i
                    self.b = self.b - self.learning_rate * ( -self.C * y_[idx] )

        return self

    def predict(self, X):
        """
        Predicts the class label for new data.
        
        Returns: labels in {-1, 1}
        """
        # Decision function: w.x + b
        linear_output = np.dot(X, self.w) + self.b
        
        # Sign function gives the class label
        return np.sign(linear_output)

# # --- Example Usage ---
# if __name__ == '__main__':
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
#     from sklearn.datasets import make_blobs
    
#     # 1. Generate linearly separable dummy data
#     X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    
#     # IMPORTANT: Ensure labels are -1 and 1
#     y = np.where(y == 0, -1, 1)
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # 2. Train the Custom SVM
#     svm = CustomSVM(learning_rate=0.001, C=0.01, n_epochs=500)
#     svm.fit(X_train, y_train)
    
#     # 3. Make predictions
#     predictions = svm.predict(X_test)
    
#     # 4. Evaluate
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"Custom SVM Accuracy: {accuracy*100:.2f}%")
#     print(f"Optimal Weights (w): {svm.w}")
#     print(f"Optimal Bias (b): {svm.b}")
    
#     # --- Visualization (requires matplotlib) ---
#     import matplotlib.pyplot as plt

#     def visualize_svm():
#         plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, s=50)
        
#         # Plot the decision boundary (w1*x1 + w2*x2 + b = 0)
#         # x2 = (-w1/w2)*x1 - (b/w2)
#         x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
#         x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

#         x1 = np.linspace(x1_min, x1_max, 100)
#         x2_decision = (-svm.w[0] / svm.w[1]) * x1 - (svm.b / svm.w[1])
        
#         # Plot the margins (w.x + b = 1 and w.x + b = -1)
#         x2_margin_p = (-svm.w[0] / svm.w[1]) * x1 - (svm.b - 1) / svm.w[1]
#         x2_margin_n = (-svm.w[0] / svm.w[1]) * x1 - (svm.b + 1) / svm.w[1]

#         plt.plot(x1, x2_decision, 'k-', label='Decision Boundary')
#         plt.plot(x1, x2_margin_p, 'k--', label='Margin (+1)')
#         plt.plot(x1, x2_margin_n, 'k--', label='Margin (-1)')
        
#         plt.title("Custom Linear SVM Decision Boundary")
#         plt.xlabel("Feature 1")
#         plt.ylabel("Feature 2")
#         plt.legend()
#         plt.show()

#     visualize_svm()

def rbf_kernel(x1, x2, gamma):
    """
    Computes the RBF kernel (Gaussian kernel) between two vectors.
    
    K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
    """
    # np.linalg.norm computes the Euclidean distance (L2 norm)
    distance_sq = np.linalg.norm(x1 - x2)**2
    return np.exp(-gamma * distance_sq)




class RBFSampler:
    """
    Approximates the RBF kernel using Random Fourier Features.
    """
    def __init__(self, n_components, gamma):
        self.n_components = n_components
        self.gamma = gamma
        self.omega = None  # Random weights
        self.b = None      # Random bias

    def fit(self, X):
        """
        Generate the random projection weights (omega) and bias (b).
        """
        n_features = X.shape[1]
        
        # 1. Sample random weights 'omega'
        #    Sampled from a Gaussian distribution N(0, 2*gamma)
        self.omega = np.random.normal(
            scale=np.sqrt(2 * self.gamma),
            size=(n_features, self.n_components)
        )
        
        # 2. Sample random bias 'b'
        #    Sampled from a Uniform distribution [0, 2*pi]
        self.b = np.random.uniform(0, 2 * np.pi, size=self.n_components)
        
        return self

    def transform(self, X):
        """
        Apply the explicit feature map to X.
        """
        if self.omega is None:
            raise RuntimeError("Sampler must be fitted first.")
            
        # 1. Project data: X @ omega + b
        projection = X.dot(self.omega) + self.b
        
        # 2. Apply cosine transformation and scale
        #    phi(X) = sqrt(2 / n_components) * cos(projection)
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)
    

# We'll use sklearn's RBF kernel calculator for convenience
from sklearn.metrics.pairwise import rbf_kernel

class NystromSampler:
    """
    Approximates a kernel map using the Nystrom method.
    """
    def __init__(self, kernel, n_components):
        # kernel: a function (like rbf_kernel)
        self.kernel = kernel 
        self.n_components = n_components
        self.basis_vectors = None
        self.normalization = None

    def fit(self, X):
        """
        Samples basis vectors and computes the normalization matrix.
        """
        n_samples = X.shape[0]
        
        # 1. Randomly sample 'n_components' from X
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.basis_vectors = X[indices]
        
        # 2. Compute the kernel matrix W (K_mm) for the basis vectors
        W = self.kernel(self.basis_vectors, self.basis_vectors)
        
        # 3. Compute SVD for W = U * S * Vh
        #    Add a small value for numerical stability
        W += 1e-6 * np.eye(self.n_components)
        U, S, Vh = np.linalg.svd(W)
        
        # 4. Store the normalization matrix: U * S^(-1/2)
        #    This is used to "whiten" the space
        self.normalization = U.dot(np.diag(1.0 / np.sqrt(S)))
        
        return self

    def transform(self, X):
        """
        Apply the Nystrom feature map to X.
        """
        if self.basis_vectors is None:
            raise RuntimeError("Sampler must be fitted first.")
            
        # 1. Compute kernel between new data (X) and basis (K_nm)
        K_nm = self.kernel(X, self.basis_vectors)
        
        # 2. Project the data: phi(X) = K_nm @ Normalization
        return K_nm.dot(self.normalization)