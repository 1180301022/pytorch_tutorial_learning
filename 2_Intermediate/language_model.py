import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus


'''
    Abstract:
语言模型——用来计算一个句子的概率的模型
'''

'''
    Note:
1. embedding:将词汇/图片等映射到多维的向量空间
2. self.embed = nn.Embedding(vocab_size, embed_size) 指定词嵌入
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epoch = 5
num_samples = 1000
batch_size = 20
seq_length = 30
learning_rate = 2e-3

# 加载数据
corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)   # 字典词数
num_batches = ids.size(1) // seq_length


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # 此处的embed_size是之前的 input_size
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)  # c0??   h是(h, c)二元组
        out = out.reshape(out.size(0)*out.size(1), out.size(2))  # (batch_size*seq_length, hidden_size)
        out = self.linear(out)
        return out, (h, c)


model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 截断反向传播？
# states: LSTM中的h, c
def detach(states):
    return [state.detach() for state in states]


for epoch in range(num_epoch):
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    for i in range(0, ids.size(1)-seq_length, seq_length):
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)
        # fp
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        # bp & optim
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)   # 梯度剪裁，针对梯度爆炸/消失
        optimizer.step()

        if (i+1) % 100 == 0:
            print("epoch:{}\tbatch:{}\tloss:{}".format(epoch+1, i+1, loss))

with torch.no_grad():
    with open('sample.txt', 'w') as f:
        states = (torch.zeros(num_layers, 1, hidden_size).to(device),
                  torch.zeros(num_layers, 1, hidden_size).to(device))
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            output, state = model(input, state)

            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)

            # 写入文件
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)