import tensorflow as tf
from kneed import KneeLocator
from keras.applications import ResNet50
from keras.models import Model
from sklearn.cluster import KMeans
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def extract_class_count(data_dir, min, max):
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
    num_cores = multiprocessing.cpu_count()

    # Loop over different numbers of clusters
    
    def kmeans_fit(cluster, features_train_flattened):
        print('iteration - {}'.format(str(cluster)))
        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(features_train_flattened)
        return cluster, kmeans.inertia_

    results = Parallel(n_jobs=num_cores)(delayed(kmeans_fit)(cluster, features_train_flattened) for cluster in range(min, max))
    SSE = [result[1] for result in results]


    
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
        

        
    return knees, SSE

# extract_objs_from_images('data/rawmotorcycles', 'tmp')
min, max = 15, 100

knees, SSE = extract_class_count('./tmp2', min, max)

print(SSE)
print(knees)
import matplotlib.pyplot as plt



plt.plot(range(min, len(SSE)+min), SSE, '-o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow plot')

for knee in knees:
    if knee is not None:
        plt.axvline(x=knee, color='red', linestyle='--')

plt.show()