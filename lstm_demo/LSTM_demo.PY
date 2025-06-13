import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

#神经网络大小设定
input_size = 3
hidden_size = input_size
hidden_layer = 3
seq_len = 30
batch_size = 100

class MyDataset(Dataset):
    def __init__(self, input, output):
        self.data, self.lable = input, output
        
    def __len__(self):
        return len(self.lable)
    
    def __getitem__(self, idx):
        return self.data[idx], self.lable[idx]

class RegLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_num_layers):
        super(RegLSTM, self).__init__()
        # 定义LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, hidden_num_layers, batch_first=True)

    def forward(self, x):
        x, (ht,ct) = self.lstm(x)
        return x
    
def to_cpu(x):
    return x.cpu().detach().numpy()

#数据准备
data_x = np.zeros(((batch_size, seq_len, input_size)))
data_y = np.zeros(((batch_size, seq_len, hidden_size)))

noise_amp = 0.5
for i in range(batch_size) :
    x = np.sin(np.linspace(1, 2, hidden_size * seq_len) * 10 + i) * (1 - noise_amp)
    y = np.copy(x)
    x += np.random.rand(input_size * seq_len) * noise_amp

    x *= 0.7
    y *= 0.7

    plt.plot(x, 'r')
    plt.plot(y)
    #plt.show()
    plt.clf()

    #input_size(batch, seq_len, input_size)
    x = np.reshape(x, (seq_len, input_size)) 
    #input_size(batch, seq_len, hidden_size * D)
    y = np.reshape(y, (seq_len, hidden_size)) 


    data_x[i, :, :] = x
    data_y[i, :, :] = y


#神经网络初始化
torch.backends.cudnn.enabled = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = RegLSTM(input_size, hidden_size, hidden_layer).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-5)
loss_func = nn.MSELoss()

total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %d" % (total))

try:
    net.load_state_dict(torch.load('ckpt2/Unet.pt'))
except:
    print("无法加载模型")

#测试
for epoch in range(2000):
    my_data = MyDataset(data_x, data_y)
    dataloader = DataLoader(my_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
    
    for i, (x, y) in enumerate(dataloader):  
        x = x.to(device)
        x = x.to(torch.float32)
        y = y.to(device)
        y = y.to(torch.float32)
        y_ = net(x)
        loss = loss_func(y, y_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 1 :
        print(to_cpu(loss))

        y_ = np.reshape(to_cpu(y_[0]), (seq_len * input_size))
        detach_y = np.reshape(to_cpu(y[0]), (seq_len * input_size))

        plt.plot(y_, 'r')
        plt.plot(detach_y)
        plt.tight_layout()
        plt.savefig('epoch' + str(epoch) + '.png')
        plt.clf()

        torch.save(net.state_dict(), 'ckpt2/Unet.pt')


