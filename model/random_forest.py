import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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

