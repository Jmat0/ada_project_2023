import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('8c_df_texture copy.csv')

# Extract the texture features and reference names into separate arrays
texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
reference_names = data['filename'].values

# Reshape the texture features to a 2D array
X = np.vstack(texture_features)

# Create the target vector y with the reference names
y = reference_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)

# Train the KNN model using only the texture features
knn.fit(X_train, y_train)

# Define the indexes of the reference names for which you want to find neighbors
indexes_to_find_neighbors = [96, 39, 47, 73, 111, 99, 213, 106, 146, 255, 1346, 1122, 1766, 1147, 1593, 1761, 942, 1363, 1092, 1847, 451, 1346, 1583, 1159, 1382]  # Example: the first 25 reference names

# Iterate over the specified indexes
for i in indexes_to_find_neighbors:
    # Get the texture features for the current reference name
    watch_features = texture_features[i]
    print("Input watch:", reference_names[i])

    # Find the closest neighbors for the current watch
    closest_neighbors = knn.kneighbors([watch_features], n_neighbors=3)

    # Retrieve the indices of the closest neighbors
    neighbor_indices = closest_neighbors[1][0]

    # Retrieve the reference names of the closest neighbors
    closest_watch_ids = [reference_names[idx] for idx in neighbor_indices]

    print("Closest neighbors:", closest_watch_ids)
    print()
