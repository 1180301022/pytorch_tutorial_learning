import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms

'''
    Abstract:
pytorch中的基本操作，包括创建张量、自动求导、
加载数据集、加载和保存模型等
将每个基础操作整合到一个方法中，方便参考
'''

'''
    Note:
1. numpy array <-> torch tensor
2. 读取数据时先创建dataset，再创建dataloader（创建线程）
3. 使用torch.save(model, name)保存模型 // 使用torch.load(name)加载模型
4. model.state_dict()返回模型的 层-参数 字典
5. torch.save(model.state_dict(), name)保存模型参数 // torch.load_state_dict(torch.load(name)) 加载保存的参数
'''

# 自动求导
def auto_grad():
    # 创建张量
    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    # 线性层，输入3dim，输出2dim
    linear = nn.Linear(3, 2)
    # 平方和损失
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

    # fp->bp->op
    # 正向传播并计算损失
    pred = linear(x)
    loss = criterion(pred, y)
    print(loss.item())

    # 反向传播并更新参数
    loss.backward()
    optimizer.step()


# numpy数组<->pytorch张量
def load_data_from_numpy():
    # 创建numpy数组
    x = np.array([[1, 2, 3], [4, 5, 6]])
    # 加载
    y = torch.from_numpy(x)
    # 返回
    z = y.numpy()


# 读入数据
def input_pipeline():
    # 从文件中取得数据
    train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
    # 取得一个训练用例
    image, label = train_dataset[0]
    print(image.size())
    print(label)
    # 加载训练数据，创建线程，指定batchsize，并混洗
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    # 生成迭代器
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    # 使用数据
    for images, labels in train_loader:
        pass


# 创建并读入自定义数据
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 1.初始化文件路径、文件名
        pass

    def __getitem__(self, index):
        # 1.读取文件
        # 2.预处理数据
        # 3.返回(x, y)
        pass
Cdataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=Cdataset, batch_size=64, shuffle=True)


# 使用预训练的模型/迁移学习，并加载和储存模型
def pretrained_model_and_load_save():
    resnet = torchvision.models.resnet18(pretrained=True)
    # 仅替换加载模型的最后一层，对其进行微调
    # 将模型的参数都设置为无需求导，即bp时不被更新
    for p in resnet.parameters():
        p.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, 100)

    # 存储和加载整个模型
    torch.save(resnet, 'model.ckpt')
    loaded_model = torch.load('model.ckpt')
    # 存储和加载模型参数。推荐操作
    torch.save(resnet.state_dict(), 'params.ckpt')
    resnet.load_state_dict(torch.load('params.ckpt'))