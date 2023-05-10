!pip install numpy
!pip install keras

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set image dimensions
img_height = 224
img_width = 224

# Load image and preprocess
image_path = 'image.jpg'
img = load_img(image_path, target_size=(img_height, img_width))
img = img.convert('RGB') # convert image to RGB format
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile and train model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(train_data, epochs=10, batch_size=20, validation_data=val_data)

# Evaluate model
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy:', test_acc)