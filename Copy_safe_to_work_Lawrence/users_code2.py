from PIL import Image
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
import torch
import torch.nn as nn
from torch.autograd import Variable

################################################ Read csv of 8d and 5_data and map it by the reference
# Rename the variable "imagefile" to "Reference" in 9d_merged_df_copy
data_change = pd.read_csv("8d_merged_df copy.csv")
data2 = pd.read_csv("5_data_with_images copy.csv")
data2.rename(columns={"Reference": "filename"}, inplace=True)
# Merge the "Brand" variable from 5_data_with_images into 9d_merged_df_copy based on "filename"
merge = data_change.merge(data2[["filename", "Brand"]], on="filename", how="left")
################################################################################################################################################################################################
################################################ Input image from User
imageFileName = input("Enter the image name with absolute path: ")
myImage = Image.open(imageFileName)
#imageFileName = "user_image"
################################################################################################################################################################################################################################################################################################
# CNN Network
class ConvNet(nn.Module):
    def _init_(self, num_classes=20):
        super(ConvNet, self)._init_()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (256,3,350,350)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1) # let's try with 12 channels as our data set is quite small
        # Shape= (256,12,350,350)
        self.bn1 = nn.BatchNorm2d(num_features=12) # same number as number of channels; number of different filters or feature maps produced by that layer
        # Shape= (256,12,350,350)
        self.relu1 = nn.ReLU() # to bring non-linearity
        # Shape= (256,12,350,350)

        self.pool = nn.MaxPool2d(kernel_size=2) # reduces the height and width of convolutional output while keeping the most salient features
        # Reduce the image size be factor 2
        # Shape= (256,12,175,175)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1) # add second conv layer to apply more patterns and increase the number of channels to 20
        # Shape= (256,20,175,175)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,175,175)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,175,175)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,175,175)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,175,175)

        self.fc = nn.Linear(in_features=175 * 175 * 32, out_features=num_classes) # fully connected layer

    # Feed forward function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,175,175)

        output = output.view(-1, 32 * 175 * 175)

        output = self.fc(output)

        return output

checkpoint= torch.load('best_checkpoint.model')
model=ConvNet(num_classes=20)
model.load_state_dict(checkpoint)
model.eval() # to set dropout and batch normalisation
#Transforms
transformer=transforms.Compose([
    transforms.Resize((350,350)),
    transforms.ToTensor()])  #0-255 to 0-1, numpy to tensors

# Categories
classes = ['audemars-piguet_images', 'blancpain_images', 'breguet_images', 'breitling_images', 'bulgari_images', 'cartier_images', 'certina_images', 'chopard_images', 'girard-perregaux_images', 'hublot_images', 'iwc_images', 'jaeger-lecoultre_images', 'montblanc_images', 'omega_images', 'panerai_images', 'patek-philippe_images', 'rolex_images', 'tag-heuer_images', 'tissot_images', 'zenith_images']

# prediction function
def prediction(img_path, transformer):
    image = Image.open(img_path).convert('RGB')

    image_tensor = transformer(image).float()

    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    input = Variable(image_tensor)

    output = model(input)

    index = output.data.numpy().argmax() # category id is the one with the highest probability

    pred = classes[index]

    return pred # output is the category name

pred_dict = {}
filename = os.path.basename(imageFileName)  # Extract the file name from the path
pred_dict[filename] = prediction(imageFileName, transformer)
print(pred_dict)
brand = pred_dict[filename]
# If the user wants to have the same brand, our next step is to filter the merge csv according to the value inside pred_dict



################################################################################################################################################################################################################################################








# Create an empty DataFrame with columns for the image filename, the grayscale histogram features, the color histogram features, and the texture features
df_gray = pd.DataFrame(columns=["filename", "gray_features"])
df_color = pd.DataFrame(columns=["filename", "color_features"])
df_texture = pd.DataFrame(columns=["filename", "texture_features"])

################################################ Resize and save the image
# Define the desired image size for preprocessing
image_size = (350, 350)

# Resize the image while maintaining aspect ratio
myImage.thumbnail(image_size, Image.LANCZOS)

# Convert the image to RGB mode
rgb_img = myImage.convert('RGB')

file_path = os.path.join("/Users/marcbourleau/PycharmProjects/scraping", "Image_test.png")
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
texture_props = np.array([graycoprops(glcm, 'contrast'), graycoprops(glcm, 'energy'), graycoprops(glcm, 'homogeneity'),
                          graycoprops(glcm, 'correlation')]).reshape(-1)
# Reshape the feature vectors to 1D arrays
gray_features = gray_hist.reshape(-1)
color_features = color_hist.reshape(-1)

df_gray = pd.concat([df_gray, pd.DataFrame({"filename": [imageFileName], "gray_features": [gray_features]})],
                    ignore_index=True)
df_color = pd.concat([df_color, pd.DataFrame({"filename": [imageFileName], "color_features": [color_features]})],
                     ignore_index=True)
df_texture = pd.concat([df_texture, pd.DataFrame({"filename": [imageFileName], "texture_features": [texture_props]})],
                       ignore_index=True)
merged_df = pd.merge(df_gray, df_color, on="filename")
merged_df = pd.merge(merged_df, df_texture, on="filename")

# Load the existing dataset
merge.to_csv('merge')
existing_dataset = pd.read_csv('merge')

# Concatenate the existing dataset and the newly created DataFrame
new_dataset = pd.concat([existing_dataset, merged_df], ignore_index=True)

# Save the new dataset to a CSV file
new_dataset.to_csv('9z_merged_df copie.csv', index=False)

################################################ Standardization of features values
################################################################################################ REMOVE REDUNDANT VARIABLES ################################################################################################
df = pd.read_csv('9z_merged_df copie.csv')

# Extract the features into separate arrays
gray_features = df['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
color_features = df['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
texture_features = df['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values

# Create a StandardScaler object
scaler = StandardScaler()

# Scale the individual feature arrays
gray_features_scaled = np.vstack(gray_features).astype(float)
gray_features_scaled = scaler.fit_transform(gray_features_scaled).tolist()

color_features_scaled = np.vstack(color_features).astype(float)
color_features_scaled = scaler.fit_transform(color_features_scaled).tolist()

texture_features_scaled = np.vstack(texture_features).astype(float)
texture_features_scaled = scaler.fit_transform(texture_features_scaled).tolist()

# Update the data DataFrame with the scaled features
df['gray_features'] = gray_features_scaled
df['color_features'] = color_features_scaled
df['texture_features'] = texture_features_scaled

#print(df)

df.to_csv('testdata.csv')

###################### KNN

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('testdata.csv')

# Extract the gray features and reference names into separate arrays
gray_features = data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
color_features = data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

reference_names = data['filename'].values

# Reshape the gray features to a 2D array
X_gray = np.vstack(gray_features)
X_color = np.vstack(color_features)
X_texture = np.vstack(texture_features)

# Concatenate the variables together into a matrix
X = np.concatenate((X_gray, X_color, X_texture), axis=1)

# Create the target vector y with the reference names
y = reference_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)

# Train the KNN model using only the gray features
knn.fit(X_train, y_train)

# Predict the closest neighbors for a specific watch (example)
watch_features = X[-1]  # Replace with the features of the watch you want to find neighbors for

# Transform the watch_features to a 2D array
# watch_features = np.reshape(watch_features, (1, 0))

closest_neighbors = knn.kneighbors([watch_features], n_neighbors=3)

# Predict the labels for the training data
y_train_pred = knn.predict(X_train)

# Retrieve the indices of the closest neighbors
neighbor_indices = closest_neighbors[1][0]

# Retrieve the unique identifiers of the closest neighbors
closest_watch_ids = [reference_names[idx] for idx in neighbor_indices]

print("Closest neighbors:", closest_watch_ids)

# Print the predicted labels
# print(y_train_pred)