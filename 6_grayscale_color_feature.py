import os
import cv2
import numpy as np

# Set the directory path containing the subdirectories with the images
dir_path = "/Users/lawrencejesudasan/Downloads/Watches_Images_Processed_copy"

# Create the grayscale directory if it does not exist
if not os.path.exists("grayscale"):
    os.makedirs("grayscale")

# Loop through each subdirectory in the directory path
for subdir in os.listdir(dir_path):
    # Create the grayscale subdirectory if it does not exist
    if not os.path.exists(os.path.join("grayscale", subdir)):
        os.makedirs(os.path.join("grayscale", subdir))
    # Loop through each image in the subdirectory
    for filename in os.listdir(os.path.join(dir_path, subdir)):
        # Load the image
        img = cv2.imread(os.path.join(dir_path, subdir, filename))
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Compute the color histogram features
        hist = cv2.calcHist([gray], [0], None, [64], [0, 64])
        hist = cv2.normalize(hist, hist)
        # Reshape the feature vector to a 1D array
        features = np.reshape(hist, (-1,))
        # Save the features in the grayscale directory with the same filename
        np.save(os.path.join("grayscale", filename[:-4] + ".npy"), features)
