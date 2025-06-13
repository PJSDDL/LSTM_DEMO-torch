import os
import random
from scipy.io import wavfile
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#神经网络大小设定
input_size = 1000
hidden_size = input_size
hidden_layer = 4
seq_len = 11 #input_size * seq_len必须等于音频长度

#随机读取N张图片的第channel个颜色通道，加噪声，制成数据集
#N大于图片总数时，读取所有图片
def audio_load(audio_dir, N):
    file_name = os.listdir(audio_dir)
    file_num = len(file_name)
    audio_X = np.zeros((N, seq_len, input_size))
    audio_Y = np.zeros((N, seq_len, hidden_size))
        
    audio_range = 0
    if (N < file_num):
        rnd_num = random.sample(range(0, file_num), N)
        audio_range = N
    else :
        rnd_num = range(0, file_num)
        audio_range = file_num

    #随机抽取N段音频
    for i in range(audio_range):
        index = rnd_num[i]
        img = np.load(audio_dir + file_name[index], allow_pickle=True)
        #print(np.shape(img))

        #plt.subplot(211)
        #plt.plot(img[0], 'r')
        #plt.subplot(212)
        #plt.plot(img[1], 'b')
        #plt.show()

        #resize
        audio_x = np.resize(img[1], (seq_len, input_size))
        audio_y = np.resize(img[0], (seq_len, hidden_size))
        #print(np.shape(img_l))

        #归一化
        audio_x = audio_x / 65535
        audio_y = audio_y / 65535

        audio_X[i] = audio_x
        audio_Y[i] = audio_y

    return audio_X.astype(np.float32), audio_Y.astype(np.float32)

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

def audio_view(x, l):
    x_ = x.cpu().detach().numpy()
    x_ = np.reshape(x_, (input_size * seq_len))
    return x_[0: l]

input_file = "train_data.wav"  
sample_rate, x_data = wavfile.read(input_file)
print('sample_rate', sample_rate)

#神经网络初始化
torch.backends.cudnn.enabled = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = RegLSTM(input_size, hidden_size, hidden_layer).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
loss_func = nn.MSELoss()

total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %d" % (total))

try:
    net.load_state_dict(torch.load('ckpt2/Unet.pt'))
except:
    print("无法加载模型")

#测试
for epoch in tqdm(range(1000 * 100)):
    data_x, data_y = audio_load('npy/', 8)
    my_data = MyDataset(data_x, data_y)
    dataloader = DataLoader(my_data, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    
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

    if epoch % 1000 == 1 :
        print('\nloss ', to_cpu(loss))
        out_wav = np.zeros((1))
        ref_wav = np.zeros((1))
        in_wav = np.zeros((1))

        def audio_stack(out, piece) :
            temp = audio_view(piece, input_size * seq_len)
            return np.concatenate((out, temp), axis=0)
        def audio_save(f_name, o):
            o = o * 65535
            o = o.astype(np.int16)
            wavfile.write(f_name, sample_rate, o)

        #画出波形图
        for i in range(4) :
            plt.subplot(211)
            plt.plot(audio_view(x[i], 100), 'k', label = 'input')
            plt.plot(audio_view(y[i], 100), 'r', label = 'truth')
            plt.plot(audio_view(y_[i], 100), label = 'output')
            plt.legend()
            plt.subplot(212)
            plt.plot(audio_view(x[i], 1000), 'k')
            plt.plot(audio_view(y[i], 1000), 'r')
            plt.plot(audio_view(y_[i], 1000))
            plt.tight_layout()
            png_name = 'epoch' + str(epoch) + 'n' + str(i) + '.png'
            plt.savefig('result/' + png_name)
            plt.clf()

            out_wav = audio_stack(out_wav, y_[i])
            ref_wav = audio_stack(ref_wav, y[i])
            in_wav = audio_stack(in_wav, x[i])

        #输出试听音频
        #print(len(out_wav))
        audio_save('result/' + 'epoch'+str(epoch)+'out.wav', out_wav)
        audio_save('result/' + 'epoch'+str(epoch)+'ref.wav', ref_wav)
        audio_save('result/' + 'epoch'+str(epoch)+'in.wav', in_wav)

        torch.save(net.state_dict(), 'ckpt2/Unet.pt')


