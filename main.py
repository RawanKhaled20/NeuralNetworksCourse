import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from skimage.feature import hog
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import datasets

# ================= LOAD FULL MNIST =================

print("Loading MNIST dataset...")

train_dataset_full = datasets.MNIST(root='./data', train=True, download=True)
test_dataset_full = datasets.MNIST(root='./data', train=False, download=True)

# Convert to numpy arrays
train_images_full = train_dataset_full.data.numpy()
train_labels_full = train_dataset_full.targets.numpy()
test_images_full = test_dataset_full.data.numpy()
test_labels_full = test_dataset_full.targets.numpy()

import matplotlib.pyplot as plt
import random

def show_random_mnist(images, labels, num_images=2):
    """
    Display random MNIST images with their labels
    """
    indices = random.sample(range(len(images)), num_images)

    plt.figure(figsize=(5, 3))

    for i, idx in enumerate(indices):
        img = images[idx]
        label = labels[idx]

        plt.subplot(1, num_images, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
show_random_mnist(train_images_full, train_labels_full, num_images=2)

# ================= BALANCED SUBSAMPLING =================
# Training: 1000 examples per digit (10,000 total)
# Testing: 200 examples per digit (2,000 total)

def balanced_subsample(images, labels, samples_per_class):
    """
    Extract balanced subsample with specified number of samples per class
    """
    balanced_images = []
    balanced_labels = []

    for digit in range(10):  # digits 0-9
        # Get indices for current digit
        digit_indices = np.where(labels == digit)[0]

        # Randomly select samples_per_class images
        selected_indices = np.random.choice(digit_indices, samples_per_class, replace=False)

        # Add to balanced dataset
        balanced_images.append(images[selected_indices])
        balanced_labels.append(labels[selected_indices])

    # Combine all digits
    balanced_images = np.vstack(balanced_images)
    balanced_labels = np.hstack(balanced_labels)

    return balanced_images, balanced_labels


print("Creating balanced subsamples...")
train_images, train_labels = balanced_subsample(train_images_full, train_labels_full, 1000)
test_images, test_labels = balanced_subsample(test_images_full, test_labels_full, 200)
show_random_mnist(train_images, train_labels, 2)

print(f"Train set shape: {train_images.shape} (should be 10000, 28, 28)")
print(f"Test set shape: {test_images.shape} (should be 2000, 28, 28)")

# Verify balanced distribution
print("\nTraining set distribution:")
for digit in range(10):
    count = np.sum(train_labels == digit)
    print(f"Digit {digit}: {count} samples")

print("\nTest set distribution:")
for digit in range(10):
    count = np.sum(test_labels == digit)
    print(f"Digit {digit}: {count} samples")

# ================= NORMALIZATION =================
# Scale pixels from [0,255] → [0,1]

train_images = train_images / 255.0
test_images = test_images / 255.0

# ================= FLATTEN FOR RAW PIXEL FEATURES =================
# Convert Matrix(28,28) → Vector(784) for PCA

train_images_flat = train_images.reshape(len(train_images), -1)
test_images_flat = test_images.reshape(len(test_images), -1)

print(f"\nFlattened shape: {train_images_flat.shape}")

IMG_SIZE = 28


# ================= FEATURE EXTRACTION FUNCTIONS =================

def extract_hog_features(images):
    """
    Extract HOG features from images
    Returns: array of HOG features
    """
    print("Extracting HOG features...")
    features = []
    for i, img in enumerate(images):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(images)} images")
        img = img.reshape(IMG_SIZE, IMG_SIZE)
        f = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
        features.append(f)
    return np.array(features)


def extract_dct_features(images):
    """
    Extract DCT features (225 dimensions from 15x15 top-left coefficients)
    """
    print("Extracting DCT features...")
    features = []
    for i, img in enumerate(images):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(images)} images")
        img = img.reshape(IMG_SIZE, IMG_SIZE)
        # Apply 2D DCT
        d = dct(dct(img.T, norm='ortho').T, norm='ortho')
        # Take top-left 15x15 coefficients (225 dimensions)
        features.append(d[:15, :15].flatten())
    return np.array(features)


def extract_pca_features(train_images, test_images):
    """
    Extract PCA features that preserve at least 95% of variance
    """
    print("Extracting PCA features...")
    pca = PCA(n_components=0.95)

    print("  Fitting PCA on training data...")
    train_pca = pca.fit_transform(train_images)

    print(f"  Number of components selected: {pca.n_components_}")
    print(f"  Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    print("  Transforming test data...")
    test_pca = pca.transform(test_images)

    return train_pca, test_pca


# ================= SVM CLASSIFIER =================

def run_svm(X_train, y_train, X_test, y_test, kernel, feature_name):
    """
    Train SVM and evaluate on test set
    """
    print(f"  Training SVM with {kernel} kernel...")
    start = time.time()

    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    train_time = time.time() - start

    print(f"  Predicting on test set...")
    start = time.time()
    pred = model.predict(X_test)
    pred_time = time.time() - start

    acc = accuracy_score(y_test, pred)
    total_time = train_time + pred_time

    print(f"  Accuracy: {acc:.4f}, Time: {total_time:.2f}s")

    return acc, total_time, pred


# ================= K-MEANS CLASSIFIER =================

def train_kmeans_classifier(X, y, k):
    """
    Train k-means classifier with k clusters per class
    """
    centroids = []
    labels = []
    classes = np.unique(y)

    for c in classes:
        Xc = X[y == c]
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(Xc)
        centroids.append(kmeans.cluster_centers_)
        labels += [c] * k

    return np.vstack(centroids), np.array(labels)


def predict_kmeans(centroids, labels, X_test):
    """
    Predict using k-means centroids
    """
    preds = []
    for x in X_test:
        dists = np.linalg.norm(centroids - x, axis=1)
        idx = np.argmin(dists)
        preds.append(labels[idx])
    return np.array(preds)


def run_kmeans(X_train, y_train, X_test, y_test, k, feature_name):
    """
    Train k-means classifier and evaluate
    """
    print(f"  Training K-means with {k} clusters per class...")
    start = time.time()

    centroids, labels = train_kmeans_classifier(X_train, y_train, k)
    preds = predict_kmeans(centroids, labels, X_test)

    total_time = time.time() - start

    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.4f}, Time: {total_time:.2f}s")

    return acc, total_time, preds


# ================= EXTRACT ALL FEATURES =================

print("\n" + "=" * 50)
print("FEATURE EXTRACTION")
print("=" * 50)

# Extract HOG features
print("\n--- HOG Features ---")
train_hog = extract_hog_features(train_images)
test_hog = extract_hog_features(test_images)
print(f"HOG feature dimension: {train_hog.shape[1]}")

# Extract DCT features
print("\n--- DCT Features ---")
train_dct = extract_dct_features(train_images)
test_dct = extract_dct_features(test_images)
print(f"DCT feature dimension: {train_dct.shape[1]}")

# Extract PCA features (on flattened raw pixels)
print("\n--- PCA Features ---")
train_pca, test_pca = extract_pca_features(train_images_flat, test_images_flat)
print(f"PCA feature dimension: {train_pca.shape[1]}")

# ================= SVM RESULTS =================

print("\n" + "=" * 50)
print("SVM CLASSIFIER RESULTS")
print("=" * 50)

svm_results = {}  # Store results for confusion matrix selection

print("\n--- HOG + SVM ---")
for kernel in ["linear", "rbf"]:
    acc, t, pred = run_svm(train_hog, train_labels, test_hog, test_labels, kernel, "HOG")
    svm_results[f"HOG_{kernel}"] = {"acc": acc, "time": t, "pred": pred}

print("\n--- PCA + SVM ---")
for kernel in ["linear", "rbf"]:
    acc, t, pred = run_svm(train_pca, train_labels, test_pca, test_labels, kernel, "PCA")
    svm_results[f"PCA_{kernel}"] = {"acc": acc, "time": t, "pred": pred}

print("\n--- DCT + SVM ---")
for kernel in ["linear", "rbf"]:
    acc, t, pred = run_svm(train_dct, train_labels, test_dct, test_labels, kernel, "DCT")
    svm_results[f"DCT_{kernel}"] = {"acc": acc, "time": t, "pred": pred}

# ================= K-MEANS RESULTS =================

print("\n" + "=" * 50)
print("K-MEANS CLASSIFIER RESULTS")
print("=" * 50)

k_values = [1, 4, 16, 32]
kmeans_results = {}  # Store results for confusion matrix selection

for k in k_values:
    print(f"\n--- K = {k} clusters per class ---")

    print("  HOG + K-means")
    acc, t, pred = run_kmeans(train_hog, train_labels, test_hog, test_labels, k, "HOG")
    kmeans_results[f"HOG_k{k}"] = {"acc": acc, "time": t, "pred": pred, "k": k}

    print("  PCA + K-means")
    acc, t, pred = run_kmeans(train_pca, train_labels, test_pca, test_labels, k, "PCA")
    kmeans_results[f"PCA_k{k}"] = {"acc": acc, "time": t, "pred": pred, "k": k}

    print("  DCT + K-means")
    acc, t, pred = run_kmeans(train_dct, train_labels, test_dct, test_labels, k, "DCT")
    kmeans_results[f"DCT_k{k}"] = {"acc": acc, "time": t, "pred": pred, "k": k}

# ================= CONFUSION MATRICES =================

print("\n" + "=" * 50)
print("CONFUSION MATRICES - BEST MODELS")
print("=" * 50)

# Find best SVM model
best_svm = max(svm_results.items(), key=lambda x: x[1]["acc"])
print(f"\nBest SVM Model: {best_svm[0]}")
print(f"Accuracy: {best_svm[1]['acc']:.4f}, Time: {best_svm[1]['time']:.2f}s")

# Plot confusion matrix for best SVM
plt.figure(figsize=(8, 6))
cm_svm = confusion_matrix(test_labels, best_svm[1]['pred'])
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title(f"Confusion Matrix - Best SVM Model: {best_svm[0]}\nAccuracy: {best_svm[1]['acc']:.4f}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Find best K-means model
best_kmeans = max(kmeans_results.items(), key=lambda x: x[1]["acc"])
print(f"\nBest K-means Model: {best_kmeans[0]}")
print(f"Accuracy: {best_kmeans[1]['acc']:.4f}, Time: {best_kmeans[1]['time']:.2f}s")
print(f"Clusters per class: {best_kmeans[1]['k']}")

# Plot confusion matrix for best K-means
plt.figure(figsize=(8, 6))
cm_kmeans = confusion_matrix(test_labels, best_kmeans[1]['pred'])
sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title(f"Confusion Matrix - Best K-means Model: {best_kmeans[0]}\nAccuracy: {best_kmeans[1]['acc']:.4f}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ================= SUMMARY STATISTICS =================

print("\n" + "=" * 50)
print("SUMMARY OF BEST RESULTS")
print("=" * 50)

print("\nTop 5 SVM Results:")
sorted_svm = sorted(svm_results.items(), key=lambda x: x[1]["acc"], reverse=True)
for i, (model, result) in enumerate(sorted_svm[:5]):
    print(f"{i + 1}. {model}: Accuracy = {result['acc']:.4f}, Time = {result['time']:.2f}s")

print("\nTop 5 K-means Results:")
sorted_kmeans = sorted(kmeans_results.items(), key=lambda x: x[1]["acc"], reverse=True)
for i, (model, result) in enumerate(sorted_kmeans[:5]):
    print(f"{i + 1}. {model}: Accuracy = {result['acc']:.4f}, Time = {result['time']:.2f}s")

print("\n" + "=" * 50)
print("CODE EXECUTION COMPLETE")
print("=" * 50)

