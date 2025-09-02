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
train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=0)

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
        x = self.linear4(x)  # 输出 logits，不加 Sigmoid
        return x

model = Model()

# 损失函数和优化器
criterion = torch.nn.BCEWithLogitsLoss()  # 推荐用这个，数值稳定
optimizer1 = torch.optim.RMSprop(model.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)

# RMSprop 训练
loss_list1 = []
for epoch in range(100):
    for i, (inputs, labels) in enumerate(train_loader):
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        loss_list1.append(loss.item())

# Adam 训练
loss_list2 = []
for epoch in range(100):
    for i, (inputs, labels) in enumerate(train_loader):
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        loss_list2.append(loss.item())

# 绘图
plt.plot(loss_list1, linestyle='dotted', label='RMSPROP')
plt.plot(loss_list2, linestyle='-', label='Adam')
plt.legend()
plt.show()
