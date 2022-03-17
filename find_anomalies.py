# import the necessary packages
import math
from functools import reduce

from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input image dataset file")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained autoencoder")
ap.add_argument("-q", "--quantile", type=float, default=0.999,
	help="q-th quantile used to identify outliers")
args = vars(ap.parse_args())

# load the model and image data from disk
print("[INFO] loading autoencoder and image data...")
autoencoder = load_model(args["model"])
images = pickle.loads(open(args["dataset"], "rb").read())
# make predictions on our image data and initialize our list of
# reconstruction errors
decoded = autoencoder.predict(images)
errors = []
# loop over all original images and their corresponding
# reconstructions
for (image, recon) in zip(images, decoded):
	# compute the mean squared error between the ground-truth image
	# and the reconstructed image, then add it to our list of errors
	mse = np.mean((image - recon) ** 2)
	errors.append(mse)
errors_sorted = np.argsort(errors)[::-1]

es = sorted(errors)

# false-positive/ false-negative
all_checks = int(len(images) / 2)

as_anomaly_classified = errors_sorted[:all_checks]
as_noanomaly_classified = errors_sorted[all_checks:]

# TODO: odległość Hamminga (różne metody zrobienia 1D z 2D)
# TODO: informacja wzajemna


def is_anomaly(v):
	if v < all_checks:
		return 1
	else:
		return 0


def not_anomaly(v):
	if v >= all_checks:
		return 1
	else:
		return 0


false_positive = reduce(lambda a, b: a + not_anomaly(b), as_anomaly_classified, 0)
false_negative = reduce(lambda a, b: a + is_anomaly(b), as_noanomaly_classified, 0)
false_positive_pc = false_positive / all_checks * 100
false_negative_pc = false_negative / all_checks * 100

# compute the q-th quantile of the errors which serves as our
# threshold to identify anomalies -- any data point that our model
# reconstructed with > threshold error will be marked as an outlier
#thresh = np.quantile(errors, args["quantile"])
#idxs = np.where(np.array(errors) >= thresh)[0]
idxs = errors_sorted[0:100]
#print("[INFO] mse threshold: {}".format(thresh))
#print("[INFO] {} outliers found".format(len(idxs)))

# initialize the outputs array
outputs2 = None
# loop over the indexes of images with a high mean squared error term
r = math.ceil(math.sqrt(errors_sorted.shape[0]))
for nj in range(0, r):
	outputs = None
	for ni in range(0, r):
		ind = nj * r + ni
		if ind >= errors_sorted.shape[0]:
			original = np.full(images[i].shape, 0)
			recon = original
		else:
			i = errors_sorted[nj * r + ni]
			# grab the original image and reconstructed image
			original = (images[i] * 255).astype("uint8")
			recon = (decoded[i] * 255).astype("uint8")
		# stack the original and reconstructed image side-by-side
		output = np.hstack([original, recon])
		# if the outputs array is empty, initialize it as the current
		# side-by-side image display
		if outputs is None:
			outputs = output
		# otherwise, vertically stack the outputs
		else:
			outputs = np.vstack([outputs, output])
	if outputs2 is None:
		outputs2 = outputs
	# otherwise, vertically stack the outputs
	else:
		outputs2 = np.hstack([outputs2, outputs])
# show the output visualization
#cv2.imshow("Output", outputs2)
#cv2.waitKey(0)
cv2.imwrite("old/test.png", outputs2)
