import torch
import numpy as np

class NEONTrainDataset(torch.utils.data.Dataset):

  def __init__(self, train_path):
    with open(train_path, 'rb') as f:
      self.images = np.load(f, allow_pickle=True)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    return self.images[idx]["rgb"].transpose([2, 0, 1]).astype("float32"), self.images[idx]["chm"].astype("float32")
