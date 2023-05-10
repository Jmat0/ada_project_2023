import cv2
import numpy as np

# Load the image
img = cv2.imread('/Users/lawrencejesudasan/Downloads/Watches_Images_Processed_copy/breguet_images/processed_9088BR-52-964-DD0D.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute the color histogram features
hist = cv2.calcHist([gray], [0], None, [64], [0, 64])
hist = cv2.normalize(hist, hist)

# Reshape the feature vector to a 1D array
features = np.reshape(hist, (-1,))
print(features)

