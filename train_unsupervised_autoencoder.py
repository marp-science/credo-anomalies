# set the matplotlib backend so figures can be saved in the background

import matplotlib
# import the necessary packages
from tensorflow.keras.models import load_model
from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import argparse
import cv2

from commons import *

# TODO: 4 osobne obrazki + razem (razem 5)
#
# ważyć przez ilość aktywnych pixeli
# np. kropki skłądają się z małej ilość, więc
# zawsze wychodzą bardziej podobne

# TODO: dla kanału kropek (z osobna sprawdzać dla innych kanałek tj. kropek, tracków, robaków i szukać kiedy najlepszy rezultat)
# - spróbować większe filtry np. 5x5, 7x7 itd.
# - mniej w features np. 8, 16 (nie 32, 64)
# - więcej tych warstw spróbować
# optymalizacja bayesowska, która ma powiedzięć jaki układ parametrów jest najlepszy


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
train_set, validation_set, test_set = prepare_dataset(args, augmentation=False)


# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
if USE_MODEL:
	autoencoder = load_model(args["model"])
else:
	(encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 1) if args['kind'] == 'mnist' else ConvAutoencoder.build(60, 60, 1)
	opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
	autoencoder.compile(loss="mse", optimizer=opt)
	autoencoder.summary()

	# train the convolutional autoencoder
	H = autoencoder.fit(
		train_set, train_set,
		shuffle=False,
		validation_data=(validation_set, validation_set),
		epochs=EPOCHS,
		batch_size=BS)

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

# use the convolutional autoencoder to make predictions on the
# testing images, construct the visualization, and then save it
# to disk

for df_func, df_name in zip([
	dm_func_mean, dm_func_avg_hash, dm_func_p_hash, dm_func_d_hash, dm_func_haar_hash, dm_func_db4_hash, dm_func_cr_hash, dm_func_color_hash
], ['mean', 'aHashref', 'pHashref', 'dHashref', 'wHashref_haar', 'wHashref_db4', 'crop_resistant_hashref', 'colorhash']):
	for img_set, set_names in zip([test_set], ['test']):
	#for img_set, set_names in zip([train_set, validation_set, test_set], ['train', 'validation', 'test']):
		print("[INFO] making predictions on train set...")
		decoded = autoencoder.predict(img_set)
		vis = visualize_predictions(decoded, img_set, df_func, set_names == 'test')
		cv2.imwrite("%s_vis_%s_%d_%s.png" % (args["kind"], set_names, used_seed, df_name), vis)


# serialize the image data to disk
#print("[INFO] saving image data...")
#f = open(args["dataset"], "wb")
#f.write(pickle.dumps(testOutX))
#f.close()

# serialize the autoencoder model to disk
#print("[INFO] saving autoencoder...")
if not USE_MODEL:
	autoencoder.save(args["model"], save_format="h5")
