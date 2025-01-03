# -*- coding: utf-8 -*-
"""SVM Implementation (no sklearn).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FK_v0x2yAcVDBR--nJv8RTk0NUGC4ihX
"""

##test with different C values, test with different kernels (cubic/quartic)

#libraries
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import KFold

#load the dataset
from google.colab import files
uploaded = files.upload()

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

data = pd.read_csv("adult.data.csv", names=columns, skipinitialspace=True, header=0)

data = data.replace('?', np.nan).dropna()

# Convert categorical columns to numerical
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income'
]
for col in categorical_columns:
    data[col] = data[col].astype('category').cat.codes

# Step 1: Check for low-variance columns and drop them
low_variance_cols = data.columns[data.std() < 1e-4]
print("Dropping low-variance columns:", low_variance_cols)
data = data.drop(columns=low_variance_cols)
data = data.drop_duplicates()

# Ensure all columns are numeric
data = data.apply(pd.to_numeric)

# Separate features and labels
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = data.drop(columns=['income']).values
X = scaler.fit_transform(X)
y = data['income'].values
y = np.where(y == 1, 1, 0)

# Define polynomial kernel functions
def polynomial_kernel(x1, x2, degree):
    return (1 + np.dot(x1, x2)) ** degree

# Implement SVM using cvxopt
class SVM:
    def __init__(self, kernel, C=None):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
      n_samples, n_features = X.shape

    # Compute the kernel matrix
      K = np.zeros((n_samples, n_samples))
      for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = self.kernel(X[i], X[j])

    # Add a small regularization term to the diagonal of the P matrix
      regularization_term = 1e-8  # Small value for numerical stability
      P = matrix(np.outer(y, y) * K + np.eye(n_samples) * regularization_term)
      q = matrix(-np.ones(n_samples))
      A = matrix(y.reshape(1, -1), (1, n_samples), 'd')
      b = matrix(0.0)

    # Define G and h matrices based on the value of C
      if self.C is None:
          G = matrix(-np.eye(n_samples))
          h = matrix(np.zeros(n_samples))
      else:
          G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
          h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

    # Solve the quadratic programming problem
      sol = solvers.qp(P, q, G, h, A, b)

    # Extract the Lagrange multipliers (alphas)
      alphas = np.ravel(sol['x'])

      # Identify the support vectors (those with non-zero alphas)
      support_vector_indices = np.where(alphas > 1e-5)[0]
      self.support_vectors = X[support_vector_indices]
      self.support_vector_labels = y[support_vector_indices]
      self.alphas = alphas[support_vector_indices]

      self.w = np.sum(
          self.alphas[:, None] * self.support_vector_labels[:, None] * self.support_vectors, axis=0
        )
      self.b = np.mean(
          [y_k - np.dot(self.w, x_k) for (x_k, y_k) in zip(self.support_vectors, self.support_vector_labels)]
        )

    def predict(self, X):
      raw_predictions = np.dot(X, self.w) + self.b
      #return np.where(raw_predictions >= 0, 1, 0)  # Map -1 to 0
      return np.sign(np.dot(X, self.w) + self.b)

# Cross-validation and evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy, precision, recall, f1_score = [], [], [], []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train SVM with cubic kernel
    svm = SVM(kernel=lambda x1, x2: polynomial_kernel(x1, x2, degree=1), C=.1)
    svm.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = svm.predict(X_test)

    acc = np.mean(y_pred == y_test)
    accuracy.append(acc)

print("Accuracy:", np.mean(accuracy))