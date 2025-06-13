import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# 读取音频并进行切分
input_file = "train_data.wav"  
output_file = "train_data_noise.wav"  

# 读取WAV文件
sample_rate_x, x_data = wavfile.read(input_file)
sample_rate_y, y_data = wavfile.read(output_file)

if not sample_rate_x == sample_rate_y :
    print('采样率不符')
    sys.exit()
sample_rate = sample_rate_x

# 声音对齐
y_data = y_data[0: len(x_data), :]

# 如果是立体声，取左声道
if len(x_data.shape) == 2:
    x_data = x_data[:, 0]
if len(y_data.shape) == 2:
    y_data = y_data[:, 0]

# 计算时间轴
duration = len(x_data) / sample_rate
time = np.linspace(0, duration, len(x_data))

# 画波形图
plt.subplot(211)
plt.plot(time, x_data, linewidth=0.5, color='blue')
plt.subplot(212)
plt.plot(time, y_data, linewidth=0.5, color='blue')
plt.tight_layout()
plt.show()

#拆分
piece_len = int(sample_rate / 4)
piece_num = int(len(x_data) / piece_len)
for i in range(piece_num) :
    piece_x = x_data[piece_len*i: piece_len*i+piece_len]
    piece_y = y_data[piece_len*i: piece_len*i+piece_len]

    plt.subplot(211)
    plt.plot(piece_x, linewidth=0.5, color='blue')
    plt.subplot(212)
    plt.plot(piece_y, linewidth=0.5, color='blue')
    plt.tight_layout()
    #plt.show()
    plt.clf()

    # 保存为试听文件
    wavfile.write('piece_wave/x'+str(i)+'.wav', sample_rate, piece_x)
    wavfile.write('piece_wave/y'+ str(i)+'.wav', sample_rate, piece_y)

    # 保存为数据集
    np.save('npy/'+str(i)+'.npy', [piece_x, piece_y])






