from PIL import Image
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

################################################ Input image from User
imageFileName = input("Enter the image name with absolute path: ")
myImage = Image.open(imageFileName)
imageFileName = "user_image"

# Create an empty DataFrame with columns for the image filename, the grayscale histogram features, the color histogram features, and the texture features
df_gray = pd.DataFrame(columns=["filename", "gray_features"])
df_color = pd.DataFrame(columns=["filename", "color_features"])
df_texture = pd.DataFrame(columns=["filename", "texture_features"])

################################################ Resize and save the image
# Define the desired image size for preprocessing
image_size = (350, 350)

# Resize the image while maintaining aspect ratio
myImage.thumbnail(image_size, Image.LANCZOS)

# Convert the image to RGB mode
rgb_img = myImage.convert('RGB')

file_path = os.path.join("/Users/lawrencejesudasan/Documents/GitHub/ADA_Project/AP", "Image_test.png")
rgb_img.save(file_path)

################################################ Extract features values
# Convert the image to grayscale
gray = rgb_img.convert('L')
# Compute the grayscale histogram features
gray_hist = np.array(gray.histogram())
gray_hist = gray_hist / np.sum(gray_hist)  # Normalize the histogram so that the values sum to 1
# Compute the color histogram features
color_hist = np.array(rgb_img.histogram())
color_hist = color_hist / np.sum(color_hist)  # Normalize the histogram so that the values sum to 1
# Compute the texture features using graycomatrix and graycoprops
gray_arr = np.array(gray)
glcm = graycomatrix(gray_arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
texture_props = np.array([graycoprops(glcm, 'contrast'), graycoprops(glcm, 'energy'), graycoprops(glcm, 'homogeneity'),
                          graycoprops(glcm, 'correlation')]).reshape(-1)
# Reshape the feature vectors to 1D arrays
gray_features = gray_hist.reshape(-1)
color_features = color_hist.reshape(-1)

df_gray = pd.concat([df_gray, pd.DataFrame({"filename": [imageFileName], "gray_features": [gray_features]})],
                    ignore_index=True)
df_color = pd.concat([df_color, pd.DataFrame({"filename": [imageFileName], "color_features": [color_features]})],
                     ignore_index=True)
df_texture = pd.concat([df_texture, pd.DataFrame({"filename": [imageFileName], "texture_features": [texture_props]})],
                       ignore_index=True)
merged_df = pd.merge(df_gray, df_color, on="filename")
merged_df = pd.merge(merged_df, df_texture, on="filename")
# Load the existing dataset
existing_dataset = pd.read_csv('8d_merged_df copy 2.csv')

# Concatenate the existing dataset and the newly created DataFrame
new_dataset = pd.concat([existing_dataset, merged_df], ignore_index=True)

# Save the new dataset to a CSV file
new_dataset.to_csv('9z_merged_df copy.csv', index=False)

################################################ Standardization of features values

df = pd.read_csv('9z_merged_df copy.csv')

# Extract the features into separate arrays
gray_features = df['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
color_features = df['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
texture_features = df['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values

# Create a StandardScaler object
scaler = StandardScaler()

# Scale the individual feature arrays
gray_features_scaled = np.vstack(gray_features).astype(float)
gray_features_scaled = scaler.fit_transform(gray_features_scaled).tolist()

color_features_scaled = np.vstack(color_features).astype(float)
color_features_scaled = scaler.fit_transform(color_features_scaled).tolist()

texture_features_scaled = np.vstack(texture_features).astype(float)
texture_features_scaled = scaler.fit_transform(texture_features_scaled).tolist()

# Update the data DataFrame with the scaled features
df['gray_features'] = gray_features_scaled
df['color_features'] = color_features_scaled
df['texture_features'] = texture_features_scaled

print(df)

df.to_csv('testdata')

###################### KNN

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('9z_merged_df copy.csv')

# Extract the gray features and reference names into separate arrays
gray_features = data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
color_features = data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

reference_names = data['filename'].values

# Reshape the gray features to a 2D array
X_gray = np.vstack(gray_features)
X_color = np.vstack(color_features)
X_texture = np.vstack(texture_features)

# Concatenate the variables together into a matrix
X = np.concatenate((X_gray, X_color, X_texture), axis=1)

# Create the target vector y with the reference names
y = reference_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)

# Train the KNN model using only the gray features
knn.fit(X_train, y_train)

# Predict the closest neighbors for a specific watch (example)
watch_features = X[-1]  # Replace with the features of the watch you want to find neighbors for

# Transform the watch_features to a 2D array
# watch_features = np.reshape(watch_features, (1, 0))

closest_neighbors = knn.kneighbors([watch_features], n_neighbors=3)

# Predict the labels for the training data
y_train_pred = knn.predict(X_train)

# Retrieve the indices of the closest neighbors
neighbor_indices = closest_neighbors[1][0]

# Retrieve the unique identifiers of the closest neighbors
closest_watch_ids = [reference_names[idx] for idx in neighbor_indices]

print("Closest neighbors:", closest_watch_ids)

# Print the predicted labels
# print(y_train_pred)