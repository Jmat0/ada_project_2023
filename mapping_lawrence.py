import os
import pandas as pd
import re

# Load the characteristics dataset
char_df = pd.read_csv('cleaned_watch_text_def.csv')
pattern = r' \(aka.*\)'
char_df['Reference'] = char_df['Reference'].apply(lambda x: re.sub(pattern, '', x))
char_df['Reference'] = char_df['Reference'].str.replace('/', '-')

# Create a new column with the image file names
char_df['image_file'] = char_df['Reference'].apply(lambda x: f"{x}.jpg")

# Define the directory where the image files are stored
image_dir = '/Users/lawrencejesudasan/Downloads/Watches_Images'

# Create a dictionary to map the characteristics to the images
char_to_image = {}
for subdir, _, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(subdir, file)
            reference = file[:-4]  # Remove the file extension
            if reference in char_df['Reference'].values:
                char_to_image[reference] = char_df.loc[char_df['Reference'] == reference].iloc[0].to_dict()

mapped_df = pd.DataFrame.from_dict(char_to_image, orient='index')

mapped_df.to_csv("data_with_images.csv", index=False)

# Check if the number of files in the image directory matches the number of rows in the characteristics dataset
#if len(os.listdir(image_dir)) != len(char_df):
    # Remove the extra rows in the characteristics dataset
    #char_df = char_df[char_df['Reference'].isin(char_to_image.keys())]

#char_df.to_csv("data_with_images.csv", index=False)

# Print the dictionary to verify the mapping
print(char_to_image)