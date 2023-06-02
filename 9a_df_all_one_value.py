import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('/Users/julian/Pycharm/AP/Project/9a_merged_df copie.csv')


#### A checker: Convertion de string a integer + structure du vector !!!!

# Extract the features into separate arrays
gray_features = data['gray_features'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
color_features = data['color_features'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
texture_features = data['texture_features'].apply(lambda x: pd.to_numeric(x, errors='coerce'))

#color_features = data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
#texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values

print("Shapes:")
print("gray_features shape:", gray_features.shape)
print("color_features shape:", color_features.shape)
print("texture_features shape:", texture_features.shape)

print("gray_features data type:", gray_features.dtype)

# Create a StandardScaler object
scaler = StandardScaler()

# Scale the individual feature arrays
gray_features_scaled = scaler.fit_transform(gray_features.astype(float).reshape(-1, 1)).reshape(-1)
color_features_scaled = scaler.fit_transform(color_features.astype(float).reshape(-1, 1)).reshape(-1)
texture_features_scaled = scaler.fit_transform(texture_features.astype(float).reshape(-1, 1)).reshape(-1)


# Update the data DataFrame with the scaled features
data['gray_features'] = gray_features_scaled.tolist()
data['color_features'] = color_features_scaled.tolist()
data['texture_features'] = texture_features_scaled.tolist()

# Save the updated DataFrame to a CSV file
data.to_csv("test.csv", index=False)
