# Install Libraries

# Import Libraries
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load Dataset
def get_image_paths(dataset_path=r"C:\Users\priya\Documents\split_dataset_new_new\train"):
    categories = os.listdir(dataset_path)  # Get all category folders
    data, labels = [], []
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):  # Ensure it's a directory
            for img_file in os.listdir(category_path):  # Get all image files in each category
                if img_file.endswith(".tif"):  # Only include TIFF files
                    img_path = os.path.join(category_path, img_file)
                    data.append(img_path)
                    labels.append(category)  # Use folder name as label
    return data, labels

data, labels = get_image_paths()
train_data, temp_data, train_labels, temp_labels = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=42
)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=2/3, stratify=temp_labels, random_state=42
)
print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

# Extract SIFT Descriptors
def extract_sift_features(image_paths):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if img is None:
            print(f"Failed to load image: {path}")
            continue
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

train_descriptors = extract_sift_features(train_data)
if len(train_descriptors) == 0:
    raise ValueError("No descriptors found. Check your training images.")
print(f"Extracted descriptors from {len(train_descriptors)} training images.")

# Build Vocabulary
def build_vocabulary(descriptors_list, num_clusters=200):
    all_descriptors = np.vstack([d for d in descriptors_list if d is not None])  # Combine all descriptors
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_descriptors)  # Perform clustering
    return kmeans

num_clusters = 200  # Number of clusters/codewords
kmeans = build_vocabulary(train_descriptors, num_clusters)
print(f"Vocabulary built with {kmeans.n_clusters} clusters.")

# Generate Histograms
def get_bags_of_sifts(image_paths, kmeans):
    sift = cv2.SIFT_create()
    histograms = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {path}")
            continue
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(kmeans.n_clusters))  # Empty histogram for images without descriptors
    return np.array(histograms)

train_histograms = get_bags_of_sifts(train_data, kmeans)
val_histograms = get_bags_of_sifts(val_data, kmeans)
test_histograms = get_bags_of_sifts(test_data, kmeans)
print("Histograms generated successfully!")

# Train and Evaluate SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(train_histograms, train_labels)

val_predictions = svm.predict(val_histograms)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {val_accuracy}")

test_predictions = svm.predict(test_histograms)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy}")

# Accuracy vs. Codewords
codeword_values = [50, 100, 200, 300]
accuracies = []

for num_clusters in codeword_values:
    kmeans = build_vocabulary(train_descriptors, num_clusters)
    train_histograms = get_bags_of_sifts(train_data, kmeans)
    val_histograms = get_bags_of_sifts(val_data, kmeans)

    svm = SVC(kernel='linear', random_state=42)
    svm.fit(train_histograms, train_labels)
    val_predictions = svm.predict(val_histograms)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    accuracies.append(val_accuracy)

plt.plot(codeword_values, accuracies, marker='o')
plt.xlabel("Number of Codewords")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy vs. Number of Codewords")
plt.show()

# t-SNE Visualization
def visualize_keypoints_tsne(descriptors_list):
    all_descriptors = np.vstack(descriptors_list)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(all_descriptors)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=1)
    plt.title("t-SNE Visualization of SIFT Keypoints")
    plt.show()

visualize_keypoints_tsne(train_descriptors)