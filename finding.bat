FOR /L %%A IN (1,1,200) DO (
  python train_unsupervised_autoencoder.py --dataset output/images.pickle --model output/autoencoder.model --kind tracks_vs_worms -s %%A
)
