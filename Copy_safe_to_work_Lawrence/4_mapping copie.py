import os
import pandas as pd
import re

# Load the characteristics dataset
char_df = pd.read_csv('2_cleaned_watch_text_def copie.csv')
pattern = r' \(aka.*\)'
char_df['Reference'] = char_df['Reference'].apply(lambda x: re.sub(pattern, '', x))
char_df['Reference'] = char_df['Reference'].str.replace('/', '-')

# Create a new column with the image file names
char_df['image_file'] = char_df['Reference'].apply(lambda x: f"{x}.jpg")
char_df['image_file_processed'] = 'processed_' + char_df['image_file']

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

# Print the dictionary to verify the mapping
print(char_to_image)

# Load the 'data_with_images.csv' file
data_with_images = pd.read_csv('data_with_images.csv')

# Load the dataset containing the 'Price' column
prices_df = pd.read_csv('2_cleaned_watch_text_def copy.csv')  # Replace 'prices.csv' with the actual file name

# Merge the datasets based on the 'Reference' column
merged_df = pd.merge(data_with_images, prices_df[['Reference', 'Price']], on='Reference', how='left')

# Replace NaN values in the 'Price' column with "NA"
merged_df['Price'].fillna("NA", inplace=True)

# Save the merged dataset to a new CSV file
merged_df.to_csv('data_with_images_and_prices.csv', index=False)
