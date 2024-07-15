import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess images
def load_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:  # Check if the image is loaded correctly
                img = cv2.resize(img, (224, 224))  # MobileNetV2 input shape
                images.append(img)
                filenames.append(filename)
            else:
                print(f"Error loading {filename}")
    return np.array(images), filenames

# Feature extraction using MobileNetV2
def extract_features(images):
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    features = model.predict(preprocess_input(images))
    return features

# Compute cosine similarity
def compute_similarity(feature1, feature2):
    similarity = cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))
    return similarity[0][0]

# Organize similar images into folders
def organize_similar_images(similar_pairs, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for pair in similar_pairs:
        folder_name = f"{pair[0].split('.')[0]}_{pair[1].split('.')[0]}"
        folder_path_pair = os.path.join(folder_path, folder_name)
        
        if not os.path.exists(folder_path_pair):
            os.makedirs(folder_path_pair)

        img1 = cv2.imread(os.path.join('shape_images', pair[0]))
        img2 = cv2.imread(os.path.join('shape_images', pair[1]))

        if img1 is not None and img2 is not None:  # Check if images are loaded correctly
            cv2.imwrite(os.path.join(folder_path_pair, pair[0]), img1)
            cv2.imwrite(os.path.join(folder_path_pair, pair[1]), img2)
        else:
            print(f"Error writing {pair[0]} or {pair[1]}")

# Load images
folder_path = 'i'
images, filenames = load_images(folder_path)

# Extract features
features = extract_features(images)

# Pairwise similarity computation
similar_pairs = []
for i in range(len(filenames)):
    for j in range(i + 1, len(filenames)):
        similarity_score = compute_similarity(features[i], features[j])
        if similarity_score > 0.7:  # Define a similarity threshold
            similar_pairs.append((filenames[i], filenames[j]))

# Organize similar images into folders
folder_path_similar = 'h'
organize_similar_images(similar_pairs, folder_path_similar)
