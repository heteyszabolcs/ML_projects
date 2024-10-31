# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Configuration options
dimension_one = 1
dimension_two = 3

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)

dimensions = {
  0: 'Sepal Length',
  1: 'Sepal Width',
  2: 'Petal Length',
  3: 'Petal Width'
}

# Color definitions
colors = {
  0: '#b40426',
  1: '#3b4cc0',
  2: '#f2da0a',
}


legend = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

colors = list(map(lambda x: colors[x], y))
plt.scatter(X[:,dimension_one], X[:,dimension_two], c=colors)
plt.title(f"Visualizing dimensions {dimension_one} and {dimension_two}")
plt.xlabel(dimensions[dimension_one])
plt.ylabel(dimensions[dimension_two])
plt.show()


