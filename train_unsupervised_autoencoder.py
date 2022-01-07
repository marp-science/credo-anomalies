# set the matplotlib backend so figures can be saved in the background

import matplotlib
# import the necessary packages
from PIL import Image
from PIL import ImageDraw

from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2

from commons import *

img = Image.new("L", (120, 12), 0)
draw = ImageDraw.Draw(img)
draw.text((0, 0), "This is a testy", 255)
text = np.array(img)
text2 = np.expand_dims(text, axis=-1)

matplotlib.use("Agg")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path to output dataset file")
ap.add_argument("-s", "--seed", type=int, required=False, default=42, help="path to output dataset file")
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained autoencoder")
#ap.add_argument("-v", "--vis", type=str, default="recon_vis.png", help="path to output reconstruction visualization file")
#ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output plot file")
ap.add_argument("-k", "--kind", type=str, default="mnist", help="path to output plot file")
args = vars(ap.parse_args())

used_seed = args["seed"]
set_global_determinism(used_seed)
trainX, testX, testOutX = prepare_dataset(args, augmentation=True)


# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 1) if args['kind'] == 'mnist' else ConvAutoencoder.build(60, 60, 1)
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)



# train the convolutional autoencoder
H = autoencoder.fit(
	trainX, trainX,
	shuffle=False,
	validation_data=(testX, testX),
	epochs=EPOCHS,
	batch_size=BS)

# use the convolutional autoencoder to make predictions on the
# testing images, construct the visualization, and then save it
# to disk

print("[INFO] making predictions on train set...")
decoded = autoencoder.predict(trainX)
vis = visualize_predictions(decoded, trainX)
cv2.imwrite("%s_vis_train_%d.png" % (args["kind"], used_seed), vis)

print("[INFO] making predictions on test set used for fit...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite("%s_vis_%d.png" % (args["kind"], used_seed), vis)

print("[INFO] making predictions on new data...")
decoded = autoencoder.predict(testOutX)
vis = visualize_predictions(decoded, testOutX)
cv2.imwrite("%s_vis_test_%d.png" % (args["kind"], used_seed), vis)


# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("%s_plot_%d.png" % (args["kind"], used_seed))

# serialize the image data to disk
#print("[INFO] saving image data...")
#f = open(args["dataset"], "wb")
#f.write(pickle.dumps(testOutX))
#f.close()

# serialize the autoencoder model to disk
#print("[INFO] saving autoencoder...")
#autoencoder.save(args["model"], save_format="h5")
