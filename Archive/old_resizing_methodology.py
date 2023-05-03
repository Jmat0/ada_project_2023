import cv2
import os


def resize_and_pad(image, size):
    # Get the current image size
    height, width = image.shape[:2]
    # Compute the aspect ratio of the image
    aspect_ratio = width / height
    # Compute the desired width and height based on the specified size and aspect ratio
    if aspect_ratio > 1:
        # Landscape orientation
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        # Portrait orientation
        new_height = size
        new_width = int(size * aspect_ratio)
    # Resize the image while maintaining its aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # Compute the amount of padding needed to reach the desired size
    top = (size - new_height) // 2
    bottom = size - new_height - top
    left = (size - new_width) // 2
    right = size - new_width - left
    # Pad the resized image to the desired size
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return padded_image



# Define the path to your images directory
images_dir = '/Users/lawrencejesudasan/Downloads/Watches_Images/zenith_images copy'

# Define the desired image size for preprocessing
image_size = 350

# Loop through all the image files in the directory
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'): # check if the file is an image file
        # Read the image
        image = cv2.imread(os.path.join(images_dir, filename))
        # Preprocess the image using the resize_and_pad function
        preprocessed_image = resize_and_pad(image, image_size)
        # Save the preprocessed image with a new filename
        new_filename = 'preprocessed_' + filename
        cv2.imwrite(os.path.join(images_dir, new_filename), preprocessed_image)