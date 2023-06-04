import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('../../../../Documents/GitHub/ADA_Project/8a_df_gray.csv')

# Extract the gray features and reference names into separate arrays
gray_features = data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
reference_names = data['filename'].values

# Reshape the gray features to a 2D array
X = np.vstack(gray_features)

# Create the target vector y with the reference names
y = reference_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)

# Train the KNN model using only the gray features
knn.fit(X_train, y_train)

# Predict the closest neighbors for a specific watch (example)
watch_features = gray_features[0]  # Replace with the features of the watch you want to find neighbors for

# Transform the watch_features to a 2D array
#watch_features = np.reshape(watch_features, (1, 0))

closest_neighbors = knn.kneighbors([watch_features], n_neighbors=3)

# Predict the labels for the training data
y_train_pred = knn.predict(X_train)


# Retrieve the indices of the closest neighbors
neighbor_indices = closest_neighbors[1][0]

# Retrieve the unique identifiers of the closest neighbors
closest_watch_ids = [reference_names[idx] for idx in neighbor_indices]

print("Closest neighbors:", closest_watch_ids)

# Print the predicted labels
#print(y_train_pred)