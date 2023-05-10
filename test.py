import cv2
import numpy as np
import os

# Define the path to the folder containing the images
path = '/Users/lawrencejesudasan/Downloads/Watches_Images_Processed_copy'

# Define the path to the new folder for grayscale arrays
grayscale_path = os.path.join(path, 'grayscale')
if not os.path.exists(grayscale_path):
    os.makedirs(grayscale_path)

# Loop over all subdirectories in the folder
for subdir in os.listdir(path):
    subpath = os.path.join(path, subdir)
    if os.path.isdir(subpath):
        # Loop over all images in the subdirectory
        for filename in os.listdir(subpath):
            filepath = os.path.join(subpath, filename)
            # Load the image
            img = cv2.imread(filepath)
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Compute the color histogram features
            hist = cv2.calcHist([gray], [0], None, [64], [0, 64])
            hist = cv2.normalize(hist, hist)
            # Reshape the feature vector to a 1D array
            features = np.reshape(hist, (-1,))
            # Save the features to a .npy file in the grayscale folder
            savepath = os.path.join(grayscale_path, subdir, filename[:-4] + '.npy')
            np.save(savepath, features)
