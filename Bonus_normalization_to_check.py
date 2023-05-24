import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('/Users/julian/Pycharm/AP/Project/merged_df.csv')


#### A checker: Convertion de string a integer + structure du vector !!!!

# Extract the features into separate arrays
gray_features = data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
color_features = data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values

print("Shapes:")
print("gray_features shape:", gray_features.shape)
print("color_features shape:", color_features.shape)
print("texture_features shape:", texture_features.shape)

# Create a StandardScaler object
scaler = StandardScaler()

# Scale the individual feature arrays
gray_features_scaled = scaler.fit_transform(gray_features.astype(float).reshape(-1, 1)).reshape(gray_features.shape)
color_features_scaled = scaler.fit_transform(color_features.astype(float).reshape(-1, 1)).reshape(color_features.shape)
texture_features_scaled = scaler.fit_transform(texture_features.astype(float).reshape(-1, 1)).reshape(texture_features.shape)

# Update the data DataFrame with the scaled features
data['gray_features'] = gray_features_scaled.tolist()
data['color_features'] = color_features_scaled.tolist()
data['texture_features'] = texture_features_scaled.tolist()

# Save the updated DataFrame to a CSV file
data.to_csv("merged_df_scaled.csv", index=False)
