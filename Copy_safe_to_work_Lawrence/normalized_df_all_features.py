import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("df_all_features.csv")

def convert_to_array(string):
    # Remove brackets and split the string into individual values
    values = string[1:-1].split()
    # Convert the values to integers and create a NumPy array
    num_array = np.array([int(value) for value in values])
    return num_array

# Apply the conversion function to the specified column
df['gray_features'] = df['gray_features'].apply(convert_to_array)


# Extract the feature columns to be scaled
features = df[['gray_features', 'color_features', 'texture_features']]  # Adjust the column names accordingly

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Scale the feature columns
scaled_features = scaler.fit_transform(features)

# Create a new DataFrame with the scaled features
df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

# Concatenate the scaled features with the remaining columns if needed
#df_scaled = pd.concat([df_scaled, df[['other_column1', 'other_column2']]], axis=1)

# Save the scaled DataFrame to a new CSV file
df_scaled.to_csv("scaled_dataset.csv", index=False)


