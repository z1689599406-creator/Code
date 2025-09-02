import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader,Dataset


# 准备数据
class Titanic_trainset(Dataset):
    def __init__(self):
        xy = np.loadtxt(r'C:\Users\16895\Desktop\Datasets\Titanic_train.csv',delimiter=',',dtype=np.float32)
        self.len = xy.shape
        self.x_data = torch.from_numpy(xy[ : , :-1])
        self.y_data = torch.from_numpy(xy[ : , :[-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len
    

class Titanic_textset(Dataset):
    def __init__(self):
        xy = np.loadtxt(r'C:\Users\16895\Desktop\Datasets\Titanic_test.csv',delimiter=',',dtype=np.float32)
        self.len = xy.shape
        self.x_data = torch.from_numpy(xy[ : , :-1])
        self.y_data = torch.from_numpy(xy[ : , :[-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len


train_set = Titanic_trainset()
text_set = Titanic_textset()
train_loader = DataLoader(dataset= train_set, batch_size=32, shuffle=True, num_workers=0)
text_loader = DataLoader(dataset= text_set, batch_size=32, shuffle=False, num_workers=0)

# 定义网络

class LogisticRegressionModel (torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = torch.nn.Linear (8,6)
    self.linear2 = torch.nn.Linear (6,4)
    self.linear3 = torch.nn.Linear (4,1)
    self.relu = torch.nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()
  
  def forward (self, x):
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))  
    x = self.linear3(x)  # 输出 logits，不加 Sigmoid
    return x

    

model = LogisticRegressionModel ()
# 定义优化器和损失函数
criterion = torch.nn.BCELoss (size_average = False)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

loss_list = []
for epoch in range(100):
   for 