This code attempts to predict tree canopy CHM from inputted RGB images. All of the training and evaluation code is in `main.py`. This code does two main steps:

1. Preprocesses the training data, dividing big mosaics into smaller 400x400 crops. This only needs to be run once, as the new array of data will be saved as a numpy file. 
2. Trains a custom architecture called CropNet to predict CHM

Dataset:

We use the RGB and CHM images from the NEON dataset. Note that CHM images (40x40) have 10x less resolution than the RGB images (400x400), so we scale them up to the size of the RGB image before passing into our loss function.

Model Architecture:

CropNet uses U-Nets in series that can be trained in an end-to-end learning fashion. First, a pretrained, frozen tree detection model like DeepForest is used to find bounding boxes for where trees are located in the input image. Then, the part of the image defined by each bounding box is put through a smaller U-Net to produce some latent representation of each individual tree. This information is concatenated as a fourth channel to the original RGB image. This new RGB image is then passed through a larger U-Net to produce a prediction for the CHM.

Loss Function:

We use a weighted MSE loss between the predicted and ground-truth CHMs that more harshly penalizes differences in output for pixels with greater height that are part of tree, so that the model focuses on accurately predicting the tree heights in the CHM.