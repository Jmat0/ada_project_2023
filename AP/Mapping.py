import os
import pandas as pd
import re

# Load the characteristics dataset
char_df = pd.read_csv('watch_text.csv')
pattern = r' \(aka.*\)'
char_df['Reference'] = char_df['Reference'].apply(lambda x: re.sub(pattern, '', x))
char_df['Reference'] = char_df['Reference'].str.replace('/', '-')

# Create a new column with the image file names
char_df['image_file'] = char_df['Reference'].apply(lambda x: f"{x}.jpg")
char_df['image_file_processed'] = 'processed_' + char_df['image_file']

# Define the directory where the image files are stored
image_dir = '/Users/marcbourleau/Desktop/Watches_Images'

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

mapped_df.to_csv("data_with_images3.csv", index=False)

# Import the CSV file
data_with_images = pd.read_csv('data_with_images2.csv')

# Select specific variables
selected_vars = ['Brand', 'Family', 'Name', 'Reference', 'Limited', 'Glass', 'Shape', 'Diameter', 'W/R', 'Color', 'image_file', 'image_file_processed']
data_selected = data_with_images[selected_vars].copy()

# Convert categorical variables to factors
categorical_vars = ['Brand', 'Family', 'Name', 'Limited', 'Glass', 'Shape', 'Color']
for var in categorical_vars:
    data_selected[var] = pd.Categorical(data_selected[var])

# Remove the first observation
data_selected = data_selected.iloc[1:]
# Reset the index after removing the first row
data_selected.reset_index(drop=True, inplace=True)
# Save the updated data as a new CSV file
data_selected.to_csv('data_with_images.csv', index=False)
