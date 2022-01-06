# set the matplotlib backend so figures can be saved in the background

import matplotlib
# import the necessary packages
from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2

from commons import *

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

# initialize the number of epochs to train for, initial learning rate,
# and batch size



#trainX = do_augmentation(trainX)



#(trainX2, testX2) = train_test_split(trainX, test_size=0.2, random_state=42)

#trainX_ = trainX == trainX2
#testX_ = testX == testX2

used_seed = args["seed"]
#init_seed(used_seed)  # 1, 3, 42, 46 - prawie ok, 50 - I.T. :D
set_global_determinism(used_seed)
trainX, testX, testOutX = prepare_dataset(args, False)


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
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite("%s_vis_%d.png" % (args["kind"], used_seed), vis)

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
print("[INFO] saving image data...")
f = open(args["dataset"], "wb")
f.write(pickle.dumps(testOutX))
f.close()

# serialize the autoencoder model to disk
print("[INFO] saving autoencoder...")
autoencoder.save(args["model"], save_format="h5")
