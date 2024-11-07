import numpy as np
from collections import Counter

# K-NN function
def knn(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        # Calculate distances between test_point and all training points
        distances = [np.linalg.norm(test_point - x) for x in X_train]
        
        # Get the k nearest neighbors and their labels
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        # Determine the most common label among neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    
    return predictions

# Example usage
if __name__ == "__main__":
    # Training data (X_train: features, y_train: labels)
    X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # Test data
    X_test = np.array([[4, 4], [5, 5]])
    
    # Set k
    k = 3
    
    # Run K-NN
    predictions = knn(X_train, y_train, X_test, k)
    print("Predictions:", predictions)
