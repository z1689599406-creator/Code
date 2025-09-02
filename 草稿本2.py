import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 数据集
class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt(r'C:\Users\16895\Desktop\Datasets\diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

# 模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)  # 输出 logits
        return x

# 损失函数
criterion = torch.nn.BCEWithLogitsLoss()

# 1️⃣ RMSprop
model_rms = Model()
optimizer_rms = torch.optim.RMSprop(model_rms.parameters(), lr=0.001)
loss_rms = []
for epoch in range(100):
    for inputs, labels in train_loader:
        y_pred = model_rms(inputs)
        loss = criterion(y_pred, labels)
        optimizer_rms.zero_grad()
        loss.backward()
        optimizer_rms.step()
        loss_rms.append(loss.item())

# 2️⃣ SGD 无动量
model_sgd = Model()
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.001)
loss_sgd = []
for epoch in range(100):
    for inputs, labels in train_loader:
        y_pred = model_sgd(inputs)
        loss = criterion(y_pred, labels)
        optimizer_sgd.zero_grad()
        loss.backward()
        optimizer_sgd.step()
        loss_sgd.append(loss.item())

# 3️⃣ SGD + momentum
model_sgd_m = Model()
optimizer_sgd_m = torch.optim.SGD(model_sgd_m.parameters(), lr=0.001, momentum=0.9)
loss_sgd_m = []
for epoch in range(100):
    for inputs, labels in train_loader:
        y_pred = model_sgd_m(inputs)
        loss = criterion(y_pred, labels)
        optimizer_sgd_m.zero_grad()
        loss.backward()
        optimizer_sgd_m.step()
        loss_sgd_m.append(loss.item())

# 绘图
plt.plot(loss_rms, linestyle='-', label='RMSprop')
plt.plot(loss_sgd, linestyle='dotted', label='SGD')
plt.plot(loss_sgd_m, linestyle='--', label='SGD+momentum')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.show()
