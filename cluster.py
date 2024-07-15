import numpy as np
import os
from shutil import copyfile
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_features(image_folder, model):
    feature_list = []
    file_list = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            filepath = os.path.join(image_folder, filename)
            
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            feature = model.predict(img_array)
            feature_list.append(feature.flatten())
            file_list.append(filename)

    return np.array(feature_list), file_list

def find_optimal_clusters(data, max_clusters=80):
    inertia_values = []
    
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def cluster_images(feature_list, optimal_clusters):
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    labels = kmeans.fit_predict(feature_list)
    
    clusters = {i: [] for i in range(optimal_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    return clusters

def save_clusters(image_folder, clusters, file_list):
    for cluster, indices in clusters.items():
        cluster_folder = os.path.join(image_folder, f"Cluster_{cluster}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        
        for idx in indices:
            filename = file_list[idx]
            src = os.path.join(image_folder, filename)
            dst = os.path.join(cluster_folder, filename)
            copyfile(src, dst)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features from images
image_folder = "i"
features, file_list = extract_features(image_folder, base_model)

# Find optimal number of clusters using Elbow method
find_optimal_clusters(features)

# Determine clusters based on optimal number
optimal_clusters = int(input("Enter the optimal number of clusters: "))
clusters = cluster_images(features, optimal_clusters)

# Save similar images in separate folders
save_clusters(image_folder, clusters, file_list)

print("Similar images saved in separate folders based on clusters.")
