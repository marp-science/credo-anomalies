import glob
import random

from PIL import Image
from imageio import imread
import numpy as np


def load_images(src):
    images = []
    for image_path in glob.glob("%s/*.png" % src):
        image = np.asarray(Image.open(image_path).convert('L'))
        images.append(image)
    return np.asarray(images)


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
