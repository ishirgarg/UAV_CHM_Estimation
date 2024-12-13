from preprocessing import training_crops
from utils import plotting
from datasets import NeonDataset

dataset = NeonDataset.NEONDataset("NEON/training", "NEON/annotations")
crops = training_crops.generate_image_crops(dataset[0], (400, 400), (80, 80), 15)

img = dataset[0]['rgb']
plotting.visualize(dataset[0])
plotting.visualize(crops[0])