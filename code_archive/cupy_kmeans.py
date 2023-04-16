import cupy as cp
import os
from keras.applications import ResNet50
import tensorflow as tf
from keras.models import Model
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

cp.show_config()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
'./tmp2',
validation_split=0.0,
seed=42,
label_mode=None,
image_size=(224, 224),
batch_size=4)

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))


feature_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)


def batch_generator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        batch_images = dataset[i:i + batch_size]
        # Reshape the batch to match the expected input shape of ResNet50
        batch_images = np.squeeze(batch_images, axis=0)
        yield batch_images


features_train = []
for batch_images in train_ds.batch(4):
    features = feature_extractor.predict(batch_images, batch_size=batch_images.shape[0])
    features_flattened = np.reshape(features, (features.shape[0], -1))
    features_train.append(features_flattened)

features_train = np.vstack(features_train)

features_train_flattened = np.reshape(features_train, (features_train.shape[0], -1))

def kmeans(X, n_clusters, max_iter=300):
    # Initialize cluster centroids randomly
    N, D = X.shape
    centroids = cp.random.randn(n_clusters, D)

    for i in range(max_iter):
        # Compute distances between each data point and each centroid
        distances = cp.sum((X[:, cp.newaxis, :] - centroids) ** 2, axis=2)

        # Assign each data point to the closest centroid
        cluster_assignments = cp.argmin(distances, axis=1)

        # Recalculate the centroids
        for j in range(n_clusters):
            centroids[j] = cp.mean(X[cluster_assignments == j], axis=0)

    return centroids, cluster_assignments


# Fit k-means model to the data
features_train_flattened = cp.array(features_train_flattened)

import matplotlib.pyplot as plt

def calculate_elbow(features, max_clusters):
    inertias = []
    for k in range(1, max_clusters + 1):
        centroids, cluster_assignments = kmeans(features, n_clusters=k)
        inertia = cp.sum(cp.min(cp.sum((features[:, cp.newaxis, :] - centroids[cluster_assignments])**2, axis=2), axis=1))
        inertias.append(inertia.get())
        
    # Plot the inertia for each number of clusters
    plt.plot(range(1, len(inertias) + 1), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Find the "elbow" by selecting the number of clusters where the change in inertia begins to slow down
    delta_inertia = [inertias[i] - inertias[i-1] for i in range(1, len(inertias))]
    elbow = delta_inertia.index(max(delta_inertia)) + 2
    plt.vlines(x=elbow, ymin=0, ymax=inertias[elbow-1], colors='red', linestyles='dashed')
    
    return elbow

print(calculate_elbow(features_train_flattened, max_clusters=5))