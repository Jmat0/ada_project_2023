from PIL import Image
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

# Input image from User
imageFileName = input("Enter the image name with absolute path: ")
myImage = Image.open(imageFileName)

# Create an empty DataFrame with columns for the image filename, the grayscale histogram features, the color histogram features, and the texture features
df_gray = pd.DataFrame(columns=["filename", "gray_features"])
df_color = pd.DataFrame(columns=["filename", "color_features"])
df_texture = pd.DataFrame(columns=["filename", "texture_features"])

# Resize and save the image
image_size = (350, 350)
myImage.thumbnail(image_size, Image.LANCZOS)
gray = myImage.convert('L')

gray_hist = np.array(gray.histogram())
gray_hist = gray_hist / np.sum(gray_hist)
small_value = 0.00001
gray_hist[gray_hist == 0] = small_value

color_hist = np.array(myImage.histogram())
color_hist = color_hist / np.sum(color_hist)
color_hist[color_hist == 0] = small_value

gray_arr = np.array(gray)
glcm = graycomatrix(gray_arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
texture_props = np.array(
    [graycoprops(glcm, 'contrast'), graycoprops(glcm, 'energy'), graycoprops(glcm, 'homogeneity'),
     graycoprops(glcm, 'correlation')]).reshape(-1)

gray_features = gray_hist.reshape(-1)
color_features = color_hist.reshape(-1)

df_gray = pd.concat([df_gray, pd.DataFrame({"filename": [imageFileName], "gray_features": [gray_features]})], ignore_index=True)
df_color = pd.concat([df_color, pd.DataFrame({"filename": [imageFileName], "color_features": [color_features]})], ignore_index=True)
df_texture = pd.concat([df_texture, pd.DataFrame({"filename": [imageFileName], "texture_features": [texture_props]})], ignore_index=True)

merged_df = pd.merge(df_gray, df_color, on="filename")
merged_df = pd.merge(merged_df, df_texture, on="filename")

print(merged_df['gray_features'])

scaler = StandardScaler()

# Scale the feature arrays
gray_features_scaled = merged_df['gray_features']
color_features_scaled = merged_df['color_features']
texture_props_scaled = merged_df['texture_features']

# Assign the feature values to the DataFrame columns
merged_df['gray_features'] = gray_features_scaled
merged_df['color_features'] = color_features_scaled
merged_df['texture_features'] = texture_props_scaled

# Read the existing DataFrame
existing_df = pd.read_csv("9b_merged_df_scaled copy 2.csv")

# Append `merged_df` to `existing_df`
appended_df = pd.concat([existing_df, merged_df], ignore_index=True)

# Save the appended DataFrame to a CSV file
appended_df.to_csv("gradio_final.csv", index=False)
