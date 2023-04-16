import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
import numpy as np
from kneed import KneeLocator

import faiss

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


data_dir = 'data/rawmotorcycles'
output_dir = 'tmp'
batch_size = 32
image_size = (224, 224)

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Extract the feature extractor
feature_extractor = Model(inputs=model.input, outputs=model.layers[-1].output)


# Load your images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
data_dir,
validation_split=0.0,
seed=42,
label_mode=None,
image_size=(224, 224),
batch_size=32)

features_train = feature_extractor.predict(train_ds)


features_train_flattened = np.reshape(features_train, (features_train.shape[0], -1))

SSE = []

# Loop over different numbers of clusters
for cluster in range(min, max):
    print('iteration - {}'.format(str(cluster)))
    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(features_train_flattened)
    SSE.append(kmeans.inertia_)

sensitivity = [1, 3, 5, 10, 100, 200, 400]
knees = []
norm_knees = []


x = range(min, len(SSE)+min)

for s in sensitivity:
    k1 = KneeLocator(x, SSE, curve="convex", direction="increasing", S=s)
    knees.append(k1.knee)
    # kl = KneeLocator(x, SSE, curve="convex", direction="decreasing", S=s)
    # knees_dsc.append(kl.knee)
    norm_knees.append(k1.norm_knee)
    