from PIL import Image
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

################################################ Input image from User
imageFileName = input("Enter the image name with absolute path: ")

myImage = Image.open(imageFileName)


################################################ Resize and save the image
# Define the desired image size for preprocessing
image_size = (350, 350)

# Resize the image while maintaining aspect ratio
myImage.thumbnail(image_size, Image.LANCZOS)

# Convert the image to RGB mode
rgb_img = myImage.convert('RGB')

file_path = os.path.join("/Users/lawrencejesudasan/Pycharm/AP", "Image_test.png")
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
texture_props = np.array([graycoprops(glcm, 'contrast'), graycoprops(glcm, 'energy'), graycoprops(glcm, 'homogeneity'), graycoprops(glcm, 'correlation')]).reshape(-1)
# Reshape the feature vectors to 1D arrays
gray_features = gray_hist.reshape(-1)
color_features = color_hist.reshape(-1)

combined_features = np.concatenate((gray_features, color_features, texture_props))
print(combined_features)
