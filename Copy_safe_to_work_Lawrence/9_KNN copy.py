import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('df_all_features.csv')

# Extract the features and reference names into separate arrays
gray_features = data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
color_features = data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
reference_names = data['filename'].values

# Combine the features into one array
X_gray = np.vstack(gray_features)
X_color = np.vstack(color_features)
X_texture = np.vstack(texture_features)

# Combine all feature arrays horizontally
X = np.hstack((X_gray, X_color, X_texture))

# Create the target vector y with the reference names
y = reference_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model using all the features
knn.fit(X_train, y_train)

# Predict the closest neighbors for a specific watch (example)
watch_gray_features = gray_features[78]  # Replace with the gray features of the watch you want to find neighbors for
watch_color_features = color_features[78]  # Replace with the color features of the watch
watch_texture_features = texture_features[78]  # Replace with the texture features of the watch

watch_features = np.hstack((watch_gray_features, watch_color_features, watch_texture_features))

closest_neighbors = knn.kneighbors([watch_features], n_neighbors=5)

# Predict the labels for the training data
y_train_pred = knn.predict(X_train)

# Retrieve the indices of the closest neighbors
neighbor_indices = closest_neighbors[1][0]

# Retrieve the unique identifiers of the closest neighbors
closest_watch_ids = [reference_names[idx] for idx in neighbor_indices]

print("Closest neighbors:", closest_watch_ids)
