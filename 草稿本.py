import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetesDataset (Dataset):
  def __init__(self):
    xy = np.loadtxt(r'C:\Users\16895\Desktop\Datasets\diabetes.csv', delimiter=',', dtype = np.float32)
    self.len = xy.shape[0]
    self.x_data = torch.from_numpy (xy[:,:-1])
    self.y_data = torch.from_numpy (xy[:,[-1]])
  
  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]
  
  def __len__(self):
    return self.len

dataset = DiabetesDataset()
train_loader = DataLoader (dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
print(train_loader)