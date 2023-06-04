from PIL import Image
import os

# Define the path to your images directory
images_dir = '../../../../../GitHub/ADA_Project/Watches_Images'

# Define the desired image size for preprocessing
image_size = (350, 350)

# Loop through all the subdirectories in the directory
for root, dirs, files in os.walk(images_dir):
    # Loop through all the image files in the subdirectory
    for filename in files:
        # check if the file is an image file
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            img = Image.open(os.path.join(root, filename))
            # Resize the image while maintaining aspect ratio
            img.thumbnail(image_size, Image.ANTIALIAS)
            # Convert the image to RGB mode
            rgb_img = img.convert('RGB')
            # Save the resized image with a new filename
            new_filename = 'processed_' + filename
            rgb_img.save(os.path.join(root, new_filename), 'JPEG')
            # Delete the original image
            os.remove(os.path.join(root, filename))