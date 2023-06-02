from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


knn = KNeighborsClassifier(n_neighbors=3)  # Specify the number of neighbors (K)
knn.fit(X_train, y_train)  # X_train represents the feature matrix, y_train represents the target vector

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the KNN model
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)
