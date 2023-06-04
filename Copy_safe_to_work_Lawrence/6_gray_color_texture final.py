import os
from PIL import Image
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Set the directory path containing the images
dir_path = "/Users/lawrencejesudasan/Downloads/Watches_Images_Processed"

# Create an empty DataFrame with columns for the image filename, the grayscale histogram features, the color histogram features, and the texture features
df_gray = pd.DataFrame(columns=["filename", "gray_features"])
df_color = pd.DataFrame(columns=["filename", "color_features"])
df_texture = pd.DataFrame(columns=["filename", "texture_features"])


# Loop through each subfolder in the directory
for subfolder in os.listdir(dir_path):
    subfolder_path = os.path.join(dir_path, subfolder)
    # Check if the subfolder is actually a directory
    if not os.path.isdir(subfolder_path):
        continue
    # Loop through each image in the subfolder
    for filename in os.listdir(subfolder_path):
        # Ignore non-image files and the .DS_Store file
        if not filename.endswith(('.jpg', '.jpeg', '.png')) or filename == '.DS_Store':
            print(f"Skipping file: {filename}")
            continue
        # Ignore files that don't start with "processed_"
        if not filename.startswith('processed_'):
            print(f"Skipping file: {filename}")
            continue
        # Extract the reference name by removing the "processed_" prefix and file extension
        reference = os.path.splitext(filename[len('processed_'):])[0]
        # Load the image
        img = Image.open(os.path.join(subfolder_path, filename))
        # Convert the image to grayscale
        gray = img.convert('L')
        # Compute the grayscale histogram features
        gray_hist = np.array(gray.histogram())
        #gray_hist = gray_hist / np.sum(gray_hist)  # Normalize the histogram so that the values sum to 1
        # Compute the color histogram features
        color_hist = np.array(img.histogram())
        #color_hist = color_hist / np.sum(color_hist)  # Normalize the histogram so that the values sum to 1
        # Compute the texture features using graycomatrix and graycoprops
        gray_arr = np.array(gray)
        glcm = graycomatrix(gray_arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        texture_props = np.array([graycoprops(glcm, 'contrast'), graycoprops(glcm, 'energy'), graycoprops(glcm, 'homogeneity'), graycoprops(glcm, 'correlation')]).reshape(-1)
        # Reshape the feature vectors to 1D arrays
        gray_features = gray_hist.reshape(-1)
        color_features = color_hist.reshape(-1)
        # Concatenate the feature vectors
        features = np.concatenate([gray_features]),  # color_features, texture_props])
        # Add the new row to the DataFrame of new data
        df_gray = pd.concat([df_gray, pd.DataFrame({"filename": [reference], "gray_features": [gray_features]})], ignore_index=True)
        df_color = pd.concat([df_color, pd.DataFrame({"filename": [reference], "color_features": [color_features]})],ignore_index=True)
        df_texture = pd.concat([df_texture, pd.DataFrame({"filename": [reference], "texture_features": [texture_props]})], ignore_index=True)

        # Print the filename of the processed image
        print(f"Processed image: {reference}")

df_gray.to_csv("8a_df_gray copy.csv")
df_color.to_csv("8b_df_color copy.csv")
df_texture.to_csv("8c_df_texture copy.csv")

merged_df = pd.merge(df_gray, df_color, on="filename")
merged_df = pd.merge(merged_df, df_texture, on="filename")
merged_df.to_csv("9a_merged_df copy.csv")

################################################ Standardization of features values

df = pd.read_csv('gradio.csv')

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

df.to_csv('gradio.csv')

#new_dataset.to_csv('9b_merged_df_scaled copy 2.csv')