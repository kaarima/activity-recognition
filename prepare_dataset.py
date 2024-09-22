import os
import cv2
import numpy as np

labeled_data_dir = 'labeled_data'

images = []
labels = []

# Loop through each label directory and load the images
for filename in os.listdir(labeled_data_dir):
    if filename.endswith('.jpg'):  
        # Load the image
        img_path = os.path.join(labeled_data_dir, filename)
        image = cv2.imread(img_path)

        # Convert the image to grayscale and resize it
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (64, 64))  

        # Append the image and label to the lists
        images.append(resized_image)
        label = filename.split('_')[0]  
        labels.append(label)

X = np.array(images)
y = np.array(labels)

np.save('X.npy', X)
np.save('y.npy', y)

print("Dataset prepared and saved as X.npy and y.npy")
