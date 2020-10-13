import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

'''
    Abstract:
线性回归任务
使用单线性层的网络，比较简单
'''

'''
    Note:
1. 超参数列在具体代码前面，方便观察和调整
2. 更新参数之前调用optimizer.zero_grad()清空梯度，防止累加
3. 只使用tensor的数据而不构建计算图时，可以
    --with torch.no_grad():
    --tensor.data
    --tensor.detach()
'''

# 超参数
input_size = 1
output_size = 1
alpha = 1e-3
num_epochs = 60

# 训练集，均为列向量
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 创建模型，选择损失函数和优化方式
linear_regression_model = nn.Linear(input_size, output_size)
criterion = nn.MSELoss()  # 最小平方和
optimizer = torch.optim.SGD(linear_regression_model.parameters(), lr=alpha)  # 指定优化目标和学习率

for epoch in range(0, num_epochs):
    # 获取训练数据：每次迭代使用全部的训练数据
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    # 正向传播并计算损失
    outputs = linear_regression_model(inputs)
    loss = criterion(outputs, targets)
    # 反向传播并更新参数
    optimizer.zero_grad()  # 重要：将优化器置0，否则会累加梯度
    loss.backward()
    optimizer.step()

    # 每5次打印结果
    if epoch % 5 == 0:
        print(str(epoch) + ',' + str(loss))

# 训练完成，查看结果
predicted = linear_regression_model(torch.from_numpy(x_train)).detach().numpy()  # tensor.detach():截断反向传播梯度流
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

torch.save(linear_regression_model.state_dict(), 'model.ckpt')