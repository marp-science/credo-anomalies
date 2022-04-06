import math
import random as rn
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from PIL import ImageDraw
import imagehash

from settings import *


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED, fast_n_close=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
    """
    set_seeds(seed=seed)
    if fast_n_close:
        return

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # # Not working with tf v2.7 but in v2.7 it not necessary
    from tfdeterminism import patch
    #patch()
    from tensorflow.python.ops import nn
    from tensorflow.python.ops import nn_ops
    from tfdeterminism.patch import _new_bias_add_1_14
    tf.nn.bias_add = _new_bias_add_1_14  # access via public API
    nn.bias_add = _new_bias_add_1_14  # called from tf.keras.layers.convolutional.Conv
    nn_ops.bias_add = _new_bias_add_1_14  # called from tests


def draw_text(text, color=255):
    img = Image.new("L", (120, 12), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, color)
    return np.array(img)


def do_augmentation(trainX, mul=100):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    arr = []
    for image in trainX:
        image = tf.expand_dims(image, 0)
        for i in range(0, mul):
            augmented_image = data_augmentation(image)
            arr.append(augmented_image[0])
    return np.vstack([arr])


def build_unsupervised_dataset(data, labels, kind='mnist'):
    # grab all indexes of the supplied class label that are *truly*
    # that particular label, then grab the indexes of the image
    # labels that will serve as our "anomalies"
    if kind == 'mnist':
        validIdxs = np.where(labels == 1)[0]
        anomalyIdxs = np.where(labels == 3)[0]
    elif kind == 'hits_vs_artefacts':
        validIdxs = np.where(labels != 4)[0]
        anomalyIdxs = np.where(labels == 4)[0]
    elif kind == 'tracks_vs_worms':
        validIdxs = np.where(labels == 2)[0]
        anomalyIdxs = np.where(labels == 3)[0]
    elif kind == 'dots_vs_worms':
        validIdxs = np.where(labels == 1)[0]
        anomalyIdxs = np.where(labels == 3)[0]
    elif kind == 'dots_vs_tracks':
        validIdxs = np.where(labels == 1)[0]
        anomalyIdxs = np.where(labels == 2)[0]
    else:
        raise Exception('Bad kind')

    # randomly shuffle both sets of indexes
    rn.shuffle(validIdxs)
    rn.shuffle(anomalyIdxs)

    # compute the total number of anomaly data points to select
    # i = int(len(validIdxs) * contam)
    # anomalyIdxs = anomalyIdxs[:i]

    # use NumPy array indexing to extract both the valid images and
    # "anomlay" images
    validImages = data[validIdxs]
    anomalyImages = data[anomalyIdxs]

    # stack the valid images and anomaly images together to form a
    # single data matrix and then shuffle the rows
    images = np.vstack([validImages, anomalyImages])
    # images = np.vstack([validImages])
    np.random.shuffle(images)

    # return the set of images
    return np.vstack([validImages]), np.vstack([anomalyImages])


def dm_func_mean(image, recon):
    err = np.mean((image - recon) ** 2)
    return math.log2(err * 5000)


def dm_func_avg_hash(image, recon):
    image_hash = imagehash.average_hash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.average_hash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return image_hash - recon_hash


def dm_func_p_hash(image, recon):
    image_hash = imagehash.phash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.phash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return image_hash - recon_hash


def dm_func_d_hash(image, recon):
    image_hash = imagehash.dhash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.dhash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return image_hash - recon_hash


def dm_func_haar_hash(image, recon):
    image_hash = imagehash.whash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.whash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return image_hash - recon_hash


def dm_func_db4_hash(image, recon):
    image_hash = imagehash.whash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)), mode='db4')
    recon_hash = imagehash.whash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)), mode='db4')
    return image_hash - recon_hash


def dm_func_cr_hash(image, recon):
    image_hash = imagehash.crop_resistant_hash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.crop_resistant_hash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return image_hash - recon_hash


def dm_func_color_hash(image, recon):
    image_hash = imagehash.colorhash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.colorhash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return image_hash - recon_hash


def visualize_predictions(decoded, images, dm_func=dm_func_mean, marked_first_half=False, max_samples=None):
    # initialize our list of output images
    gt = images
    outputs2 = None
    samples = math.ceil(math.sqrt(gt.shape[0]))
    if max_samples is not None:
        samples = min(samples, max_samples)

    errors = []
    for (image, recon) in zip(images, decoded):
        # compute the mean squared error between the ground-truth image
        # and the reconstructed image, then add it to our list of errors
        mse = dm_func(image, recon)
        errors.append(mse)
    errors_sorted = np.argsort(errors)[::-1]

    # loop over our number of output samples
    for y in range(0, samples):
        outputs = None
        for x in range(0, samples):
            i = y * samples + x
            if i >= gt.shape[0]:
                original = np.full(gt[0].shape, 0)
                recon = original
                i_sorted = 0
            else:
                # grab the original image and reconstructed image
                i_sorted = errors_sorted[i]
                original = (gt[i_sorted] * 255).astype("uint8")
                recon = (decoded[i_sorted] * 255).astype("uint8")

            # stack the original and reconstructed image side-by-side
            output = np.hstack([original, recon])
            v = "" if i >= gt.shape[0] else '      %0.6f' % errors[errors_sorted[i]]
            color = 255
            if marked_first_half and i_sorted < gt.shape[0]/2:
                color = 128
            text = np.expand_dims(draw_text(v, color), axis=-1)
            output = np.vstack([output, text])

            # if the outputs array is empty, initialize it as the current
            # side-by-side image display
            if outputs is None:
                outputs = output

            # otherwise, vertically stack the outputs
            else:
                outputs = np.vstack([outputs, output])

        if outputs2 is None:
            outputs2 = outputs

        # otherwise, horizontally stack the outputs
        else:
            outputs2 = np.hstack([outputs2, outputs])

    # return the output images
    return outputs2, errors


def prepare_dataset(args, augmentation=False):
    if args["kind"] == "mnist":
        from tensorflow.keras.datasets import mnist
        print("[INFO] loading MNIST dataset...")
        ((train_set, trainY), (unused_set, unused_set2)) = mnist.load_data()
    else:
        from dataset_loader import load_dataset
        print("[INFO] loading CREDO dataset...")
        train_set, trainY = load_dataset()

    # build our unsupervised dataset of images with a small amount of
    # contamination (i.e., anomalies) added into it
    print("[INFO] creating unsupervised dataset...")
    images, anomalies = build_unsupervised_dataset(train_set, trainY, kind=args["kind"])

    # add a channel dimension to every image in the dataset, then scale
    # the pixel intensities to the range [0, 1]
    images = np.expand_dims(images, axis=-1)
    images = images.astype("float32") / 255.0

    anomalies = np.expand_dims(anomalies, axis=-1)
    anomalies = anomalies.astype("float32") / 255.0

    # construct the training and testing split
    (train_set, test_set) = train_test_split(images, test_size=0.2)

    if augmentation:
        train_set = do_augmentation(train_set)

    (train_set, validation_set) = train_test_split(train_set, test_size=0.2)

    # prepare test set
    max_test = min(anomalies.shape[0], test_set.shape[0])
    test_set = np.vstack([anomalies[0:max_test], test_set[0:max_test]])

    return train_set, validation_set, test_set


def original_autoencoder():
    from pyimagesearch.convautoencoder import ConvAutoencoder
    from keras.optimizer_v2.adam import Adam

    (encoder, decoder, autoencoder) = ConvAutoencoder.build(60, 60, 1)
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    autoencoder.compile(loss="mse", optimizer=opt)
    return autoencoder


def train_or_cache(train_set, autoencoder, fncache=None, force_train=False, epochs=EPOCHS, batch_size=BS, shuffle=False, validation_set=None):
    from os.path import exists
    from keras.models import load_model
    import matplotlib.pyplot as plt

    fn = fncache  # 'cache/%s.h5' % str(fncache)

    if fncache is not None and exists(fn) and not force_train:
        print('Load from: %s' % fn)
        return load_model(fn)

    #(input_set, validation_set) = train_test_split(train_set, test_size=0.2)
    # train the convolutional autoencoder
    H = autoencoder.fit(
        train_set,
        train_set,
        shuffle=shuffle,
        validation_data=(validation_set, validation_set) if validation_set is not None else None,
        epochs=epochs,
        batch_size=batch_size
    )

    if fncache is not None:
        autoencoder.save(fn, save_format="h5")
        print('Saved in: %s' % fn)

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    if validation_set is not None:
        plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(fn.replace('.h5', 'png'))

    return autoencoder


def binarize_image(image, cutoff_qt_value=0):
    """
    Binaryzacja obrazka na podstawie podanego progu. Domyślnie, co nie jest całkiem czarne, jest białe.

    :param image: źródlowy obrazek
    :param cutoff_qt_value: próg binaryzacji, 1 - większe od progu
    :return:
    """
    return np.where(image > cutoff_qt_value, 1, 0)


def cutoff_reconstruction_background(image, reconstruction):
    """
    Odcina tło od rekonstrukcji, która w oryginalnym obrazku była tłem.

    W oryginalnym obrazku mamy coś na tle czerni (0).

    :return: rekonstrukcja z obciętym tłem
    """
    return binarize_image(image, 0) * reconstruction

def count_non_black_pixels(image):
    """
    Zwraca liczbę nieczarnych pikseli z obrazka.

    :param image: obrazek.
    :return: liczba nieczarnych pikseli.
    """
    return np.count_nonzero(image)


def compute_errors(image, recon, dm_func, normalize=True):
    """
    Obliczanie błędu.

    :param image: obrazek źródłowy.
    :param recon: rekonstrukcja.
    :param dm_func: funkcja porównująca, jako parametr przyjmuje (image, recon), zwraca skalar będący miarą podobieństwa.
    :param normalize: jeśli true, to dzieli wynik dm_func przez liczbę nieczarnych pikseli z image.
    :return: tablica
    """
    return dm_func(image, recon, normalize)


def prepare_for_histogram(images, reconstructions, dm_func, normalize=True, cutoff_background=False, binarize_for_compare=False):
    errors = []
    for (image, recon) in zip(images, reconstructions):
        try:
            if cutoff_background:
                recon = cutoff_reconstruction_background(image, recon)
            if binarize_for_compare:
                image = binarize_image(image)
                recon = binarize_image(recon)
            mse = compute_errors(image, recon, dm_func, normalize)
            errors.append(mse)
        except:
            errors.append(0)
    return errors


def dm_func_mean2(image, recon, normalize=True):
    blacks = 1
    if normalize:
        blacks = count_non_black_pixels(image)
    if blacks == 0:
        blacks = 1

    err = np.mean((image - recon) ** 2) / (blacks ** 2)
    return math.log2(err * 5000)


def calc_similarity(autoencoder, dots_set, tracks_set, worms_set, artifacts_set, **argv):
    dots_reconstruction = autoencoder.predict(dots_set)
    worms_reconstruction = autoencoder.predict(worms_set)
    tracks_reconstruction = autoencoder.predict(tracks_set)
    artifacts_reconstruction = autoencoder.predict(artifacts_set)

    return {
        'dots': prepare_for_histogram(dots_set, dots_reconstruction, dm_func_mean2, **argv),
        'worms': prepare_for_histogram(worms_set, worms_reconstruction, dm_func_mean2, **argv),
        'tracks': prepare_for_histogram(tracks_set, tracks_reconstruction, dm_func_mean2, **argv),
        'artifacts': prepare_for_histogram(artifacts_set, artifacts_reconstruction, dm_func_mean2, **argv)
    }
