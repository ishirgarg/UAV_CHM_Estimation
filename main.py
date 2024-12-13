'''
Copyright (c) 2024 Ishir Garg
'''

import matplotlib.pyplot as plt
from deepforest import main as dfmain
import supervision as svn
from supervision import Detections
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from src.datasets.NeonDataset import NEONDataset
from src.datasets.NEONTrainDataset import NEONTrainDataset
import src.utils.cv as cv_utils 
import src.utils.plotting as plotting_utils
import src.preprocessing.training_crops as preprocessing
from src.models.UNet import UNet, MiniUNet

def predict_trees(rgb_img, df_model):
    '''Predicts bounding boxes for a given DeepForest model'''
    raw_detections = df_model.predict_image(image=rgb_img.astype("float32"))
    if raw_detections is None:
        return [], []
    bboxes = np.array([cv_utils.box_prediction_to_xyxy(raw_detections.iloc[i]) for i in range(len(raw_detections))])
    scores = np.array([raw_detections.iloc[i]["score"] for i in range(len(raw_detections))])
    return bboxes, scores

def preprocess_crops():
    '''Preprocess the mosaics into indiviudal images'''
    dataset = NEONDataset("NEON/training", "NEON/annotations")
    preprocessing.generate_dataset_crops(dataset, (400, 400), (80, 80), 15, "NEON_preprocessed/train")


class CropNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.individual_unet = MiniUNet()
        self.large_unet = UNet(4)

    def forward(self, x):
        new_x = torch.cat([x, torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3])).to(device)], dim=1)
        for i, img in enumerate(x):
            bboxes, _ = predict_trees(np.array(img.cpu().numpy()).transpose([1, 2, 0]), df_model)
            for box in bboxes:
                out = self.individual_unet(img[:, box[0]:box[2], box[1]:box[3]].unsqueeze(0))
                output_tensor = torch.nn.functional.interpolate(out, size=(box[2] - box[0], box[3] - box[1]), mode='bilinear')
                new_x[i][3][box[0]:box[2], box[1]:box[3]] = output_tensor

        return self.large_unet(new_x)


BATCH_SIZE = 16
EPOCHS = 1
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Only need to run this once!
# preprocess_crops()

train_data = torch.utils.data.Subset(NEONTrainDataset("NEON_preprocessed/train.npy"),
                                     (8,))
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

df_model = dfmain.deepforest()
df_model.use_release()
# vit_model = torchvision.models.vit_b_16(torchvision.models.ViT_B_16_Weights.DEFAULT)


class WeightedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_pred, img_true):
        # Ensure images are in the same shape and device
        assert img_pred.shape == img_true.shape, "Input tensors must have the same shape."
        
        # Compute weighted squared differences
        squared_diff = (img_pred - img_true) ** 2
        squared_diff = (img_true - torch.min(img_true) + 1) * squared_diff
        
        # Return mean weighted loss
        loss = torch.mean(squared_diff)
        return loss

# model = CropNet().to(device)
model = UNet(3).to(device)
model.train()
criterion = WeightedMSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

training_loss = []

for epoch in range(EPOCHS):
    print(f"Starting epoch {epoch}")
    model.train()  # Set the model to training mode

    for batch_idx, (rgb, chm) in enumerate(train_loader):
        rgb, chm = rgb.to(device), chm.to(device)

        optimizer.zero_grad()
        outputs = model(rgb)
        
        padded_chm = chm.repeat_interleave(10, dim=1)
        padded_chm = padded_chm.repeat_interleave(10, dim=2)
        loss = criterion(outputs, padded_chm.unsqueeze(1))

        #im1 = outputs.cpu().detach().numpy().squeeze(0).squeeze(0)
        #im2 = padded_chm.cpu().detach().numpy().squeeze(0)
        #plotting_utils.visualize_chms(im1, im2)

        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
        print("Loss:", training_loss[-1])
    
torch.save(model.state_dict(), "model.pt")

plt.plot(training_loss)
plt.show()

# EVAL TEST

rgb, chm = train_data[0]
trained_model = UNet(3).to(device)
trained_model.load_state_dict(torch.load("model.pt"))
trained_model.eval()

pred_chm = model(torch.Tensor([rgb]).to(device)).squeeze()
print(pred_chm.shape, chm.shape)
plotting_utils.visualize_rgb_chm(rgb.transpose([1, 2, 0]), chm)
plotting_utils.visualize_chms(pred_chm.cpu().detach().numpy(), chm)
exit(0)

# EVAL TEST END
