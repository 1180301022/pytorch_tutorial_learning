import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

'''
    Abstract:
MNIST数据集上的多分类问题
linear->softmax
输入图像，输出十种分类结果的概率
只训练linear层的参数
'''

'''
    Note:
对于a.size() = (m, n)

torch.max(a) == a.max(): 返回a中的最大值（无索引）
torch.max(a, 0) == a.max(0): 返回tensor a 每列中最大的元素和索引 (n, 1)
torch.max(a, 1) == a.max(1): 返回tensor a 每行中最大的元素和索引 (m, 1)
因为输入按行排列，所以要返回每行中最大元素的索引

torch.sum(a) == a.sum(): 返回tensor(1, 1)
torch.sum(a, 0) == a.sum(0): 返回tensor(n, 1)，按列求和
torch.sum(a, 1) == a.sum(1): 返回tensor(m, 1)，按行求和
'''

# 设置超参数
input_size = 28 * 28  # ==784
num_classes = 10  # 多分类问题
num_epochs = 5
batch_size = 100
alpha = 1e-3

# 获取数据集 MNIST，并创建dataloader
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())
# 创建dataloader要指定数据集，batch_size
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

# 使用一层线性模型
model = nn.Linear(input_size, num_classes)

# 指定代价函数和优化器
# ！！！交叉熵损失内置计算softmax
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), alpha)

# 每个epoch使用所有训练集数据
for epoch in range(0, num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):

        images = images.reshape(-1, input_size)  # batch_size x input_size

        # 正向传播
        y_pred = model(images)
        loss = criterion(y_pred, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('epoch:{}\tbatch{}\tloss{}'.format(epoch, i, loss))

# 测试模型
with torch.no_grad():
    num_correct = 0
    num_total = 0
    for images, labels in test_dataloader:
        images = images.reshape(-1, input_size)  # batch_size x input_size
        output = model(images)  # linear层的输出，没有经过softmax
        _, prediction = torch.max(output.data, 1)  # 使用tensor.data，意在仅仅使用数据，避免构建计算图
        num_total += labels.size(0)
        # 直接相等会构造一个布尔数组，目前没有找到好的转换方案，暂时用循环求解
        for i in range(0, prediction.size(0)):
            if torch.equal(prediction[i], labels[i]):
                num_correct += 1
    print('accuracy:{}'.format(num_correct/num_total))

