import os
from PIL import Image
import numpy as np
import pandas as pd

# Set the directory path containing the images
dir_path = "/Users/julian/Desktop/Watches_Images_Processed copie"

# Create an empty DataFrame with columns for the image filename and the color histogram features
df_new = pd.DataFrame(columns=["filename", "features"])

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
            continue
        # Ignore files that don't start with "processed_"
        if not filename.startswith('processed_'):
            continue
        # Extract the reference name by removing the "processed_" prefix and file extension
        reference = os.path.splitext(filename[len('processed_'):])[0]
        # Load the image
        img = Image.open(os.path.join(subfolder_path, filename))
        # Convert the image to grayscale
        gray = img.convert('L')
        # Compute the color histogram features
        hist = np.array(gray.histogram())
        hist = hist / np.sum(hist)  # Normalize the histogram so that the values sum to 1
        # Reshape the feature vector to a 1D array
        features = hist.reshape(-1)
        # Add the new row to the DataFrame of new data
        df_new = pd.concat([df_new, pd.DataFrame({"filename": [reference], "features": [features]})], ignore_index=True)

# Load your existing DataFrame from the "data_with_images.csv" file
df_existing = pd.read_csv("5_data_with_images copie.csv")

# Merge the new DataFrame with the existing DataFrame using the reference name column as the link
df_merged = pd.merge(df_existing, df_new, left_on="Reference", right_on="filename")

# Drop the redundant "filename" column
df_merged.drop("filename", axis=1, inplace=True)

# Save the merged DataFrame to the existing file
df_merged.to_csv("5_data_with_images copie.csv", index=False)
