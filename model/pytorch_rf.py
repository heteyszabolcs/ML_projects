import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load data
data = load_iris()
X, y = data.data, data.target

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert data to PyTorch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# create data loaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# KNN
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        distances = torch.cdist(X, self.X_train)
        knn_indices = distances.topk(self.k, largest=False).indices
        knn_labels = self.y_train[knn_indices]
        predictions = torch.mode(knn_labels, dim=1).values
        return predictions

# initializing and training the model
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print(f"Predictions: {predictions}")

# Calculate accuracy
accuracy = (predictions == y_test).float().mean()
print(f"Accuracy: {accuracy:.4f}")

# random forest
import torch
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        # Stop conditions for recursion
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            self.tree = torch.mode(y).values.item()  # Assign majority class
            return

        # Find the best feature and threshold to split on
        best_split = self._best_split(X, y)
        if best_split is None:
            self.tree = torch.mode(y).values.item()
            return

        # Recursive split
        feature, threshold, left_indices, right_indices = best_split
        self.tree = {
            'feature': feature,
            'threshold': threshold,
            'left': DecisionTree(self.max_depth, self.min_samples_split).fit(X[left_indices], y[left_indices],
                                                                             depth + 1),
            'right': DecisionTree(self.max_depth, self.min_samples_split).fit(X[right_indices], y[right_indices],
                                                                              depth + 1)
        }

    def _best_split(self, X, y):
        # Implement logic to find the best feature and threshold split based on Gini or entropy
        # Returns feature index, threshold, and indices for left and right splits
        # Placeholder code for illustration:
        return None

    def predict(self, X):
        # Recursive prediction based on tree structure
        if isinstance(self.tree, dict):
            if X[:, self.tree['feature']] <= self.tree['threshold']:
                return self.tree['left'].predict(X)
            else:
                return self.tree['right'].predict(X)
        else:
            return self.tree  # Return leaf node prediction

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2):
        self.n_estimators = n_estimators
        self.trees = [DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split) for _ in range(n_estimators)]

    def fit(self, X, y):
        for tree in self.trees:
            # Bootstrap sample
            indices = torch.randint(0, X.size(0), (X.size(0),))
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        # Collect predictions from each tree
        X = torch.tensor(X)
        tree_preds = torch.stack([tree.predict(X) for tree in self.trees], dim=1)
        # Majority voting
        return torch.mode(tree_preds, dim=1).values

# random forest with multiprocessing
def train_tree(tree, X, y):
    tree.fit(X, y)

class RandomForestParallel:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2):
        self.trees = [DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        with mp.Pool() as pool:
            pool.starmap(train_tree, [(tree, X, y) for tree in self.trees])

# integrating with PyTorch training pipelines
# embedding KNN/Random Foresst into training loops
class HybridModel(nn.Module):
    def __init__(self, rf_model, input_dim):
        super(HybridModel, self).__init__()
        self.rf_model = rf_model
        self.fc = nn.Linear(input_dim + 1, 10) # Neural network after random forest

    def forward(self, X):
        rf_pred = self.rf_model.predict(X).unsquezze(1).float() # Embed RF predictions
        combined_input = torch.cat((X, rf_pred), dim=1)
        return self.fc(combined_input)

# initializing and training the model
rf = RandomForest(n_estimators=10, max_depth=5, min_samples_split=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


predictions = rf.predict(X_test)
print(f"Predictions: {predictions}")

# Calculate accuracy
accuracy = (predictions == y_test).float().mean()
print(f"Accuracy: {accuracy:.4f}")

