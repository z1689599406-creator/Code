from torch.nn import ModuleList
import torch
import torch.nn.functional as F

# 准备数据
x_data = torch.Tensor ([[1.0], [2.0], [3.0]])
y_data = torch.Tensor ([[0], [0], [1]])

# 定义网络
class LogisticRegressionModel (torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear (1, 1)
  
  def forward (self, x):
    y_pred = F.sigmoid (self.linear(x))
    return y_pred

model = LogisticRegressionModel ()
# 定义优化器和损失函数
criterion = torch.nn.BCELoss (size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(1000):
  y_pred = model (x_data)
  loss = criterion (y_pred, y_data)
  print (epoch, loss.item())

  optimizer.zero_grad()
  loss.backward ()
  optimizer.step()

print ('w=', model.linear.weight.item())
print ('b=', model.linear.bias.item())

x_test = torch.Tensor ([[4.0]])
y_test = model (x_test)
print ('y_pred = ', y_test.data)


