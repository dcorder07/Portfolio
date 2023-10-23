#David Cordero
#Data Mining Assignment 2
#KNN, Decision Tree, and Naive Bayes

import numpy as np

# Load in the data sets 
train_data = np.genfromtxt('C:/Users/dcord/Downloads/spam_train.csv', delimiter=',')
test_data = np.genfromtxt('C:/Users/dcord/Downloads/spam_test.csv', delimiter=',')

# euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# knn classifier function
def knn_classifier(train_data, test_data, k):
    predicted_labels = []

    for test_instance in test_data:
        distances = []

        for train_instance in train_data:
            # distance btw test and training
            dist = euclidean_distance(test_instance[:-1], train_instance[:-1])  
            distances.append((dist, train_instance[-1]))  

        # Sort distances to find the k-nearest neighbors
        distances.sort(key=lambda x: x[0])

        # Get the k-nearest class labels
        k_nearest_labels = [label for (_, label) in distances[:k]]

        # Make a majority vote to classify the test instance
        predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predicted_labels.append(predicted_label)

    return predicted_labels

# Calculate test accuracies for different values of k without normalization (part a)
k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
for k in k_values:
    predicted_labels = knn_classifier(train_data, test_data, k)
    correct_predictions = sum(1 for true, pred in zip(test_data[:, -1], predicted_labels) if true == pred)
    accuracy = correct_predictions / len(test_data)
    print(f"k={k}, Accuracy: {accuracy}")

# Define z-score normalization function
def z_score_normalization(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data

# Normalize the train and test data (part b)
train_data[:, :-1] = z_score_normalization(train_data[:, :-1])
test_data[:, :-1] = z_score_normalization(test_data[:, :-1])

# Calculate test accuracies for different values of k with z-score normalization (part b)
for k in k_values:
    predicted_labels = knn_classifier(train_data, test_data, k)
    correct_predictions = sum(1 for true, pred in zip(test_data[:, -1], predicted_labels) if true == pred)
    accuracy = correct_predictions / len(test_data)
    print(f"k={k}, Accuracy with normalization: {accuracy}")

# Generate KNN predicted labels for the first 50 instances (part c)
for k in k_values:
    predicted_labels = knn_classifier(train_data, test_data[:50], k)
    print(f"KNN Predicted Labels for k={k}: {predicted_labels}")




