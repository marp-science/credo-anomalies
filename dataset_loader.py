import glob
import random

from PIL import Image
from imageio import imread
import numpy as np
import tensorflow as tf
from keras.layers import RandomFlip, RandomRotation


DATA_SETS = {
    'dots': "hit-images-final/hits_votes_4_Dots",
    'tracks': "hit-images-final/hits_votes_4_Lines",
    'worms': "hit-images-final/hits_votes_4_Worms",
    'artifacts': "hit-images-final/artefacts"
}


def do_binarize(image):
    img = image.astype('int32')
    blackwhite = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0

    threshold = blackwhite.mean() + blackwhite.std() * 5
    threshold = threshold if threshold < 100 else 100
    mask = np.where(blackwhite > threshold, 1, 0)
    blackwhite = blackwhite * mask
    return blackwhite


def load_images(src):
    images = []
    files = list(glob.glob("%s/*.png" % src))
    files = sorted(files)
    for image_path in files:
        #print(image_path)
        #image = np.asarray(Image.open(image_path).convert('L'))
        image = np.asarray(Image.open(image_path))
        image = do_binarize(image)
        images.append(image)
    return np.asarray(images).astype("float32") / 255.0


def do_augmentation2(images, mul=1, data_augmentation=None):
    if mul == 1:
        return images

    if data_augmentation is None:
        data_augmentation = tf.keras.Sequential([
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.2),
        ])

    arr = []
    for image in images:
        image = tf.expand_dims(image, 0)
        for i in range(0, mul):
            augmented_image = data_augmentation(image)
            arr.append(augmented_image[0])
    return np.vstack([arr])


def prepare_data(src, augmentation=1):
    images = load_images(src)
    expanded = do_augmentation2(images, augmentation)
    return expanded


def load_dataset_with_cache(dataset, augmentation=1, force_load=False):
    from os.path import exists
    import pickle

    fn = 'cache/dataset_%s_%d.pickle' % (dataset, augmentation)
    if not force_load and exists(fn):
        return pickle.loads(open(fn, "rb").read())

    images = np.expand_dims(prepare_data(DATA_SETS[dataset], 1), axis=-1)
    expanded = np.expand_dims(prepare_data(DATA_SETS[dataset], augmentation), axis=-1)
    data_set = (images, expanded)

    f = open(fn, "wb")
    f.write(pickle.dumps(data_set))
    f.close()

    return data_set


def load_dataset():
    x_spots = load_images("hit-images-final/hits_votes_4_Dots")
    x_tracks = load_images("hit-images-final/hits_votes_4_Lines")
    x_worms = load_images("hit-images-final/hits_votes_4_Worms")
    x_artifacts = load_images("hit-images-final/artefacts")

    x_all = np.vstack([x_spots, x_tracks, x_worms, x_artifacts])

    y_spots = np.full((x_spots.shape[0]), 1)
    y_tracks = np.full((x_tracks.shape[0]), 2)
    y_worms = np.full((x_worms.shape[0]), 3)
    y_artifacts = np.full((x_artifacts.shape[0]), 4)

    y_all = np.hstack([y_spots, y_tracks, y_worms, y_artifacts])
    return x_all, y_all


def load_data(seed=1, div=4):
    """
    Dataset, return based on MNIST dataset:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    """
    x_all, y_all = load_dataset()
    indices = list(range(0, x_all.shape[0]))
    random.seed(seed)
    random.shuffle(indices)
    splitter = int(len(indices) - len(indices) / div)
    train = indices[0:splitter]
    test = indices[splitter:]

    x_train = x_all[train]
    y_train = y_all[train]
    x_test = x_all[test]
    y_test = y_all[test]

    return (x_train, y_train), (x_test, y_test)
