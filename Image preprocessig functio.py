import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

def preprocess_images(folder_path, resolution):
  
    image_data = []
    labels = []

    label_mapping = {label: idx for idx, label in enumerate(os.listdir(folder_path))}

    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            label = label_mapping[class_folder]
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)

              
                image = cv2.imread(image_path)
                image = cv2.resize(image, resolution)

                image = img_to_array(image)
                image = image / 255.0

                image_data.append(image)
                labels.append(label)

  
    labels = np.array(labels)

    labels = to_categorical(labels)

    X = np.array(image_data)

    return X, labels

folder_path = "/content/drive/MyDrive/dataset-20231225T050159Z-001/dataset"
resolution = (100, 100)  
X, y = preprocess_images(folder_path, resolution)


