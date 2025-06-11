import os
import struct
import numpy as np
import scipy.fftpack as fft
from scipy.signal import get_window

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ManualMFCC:
    def __init__(self, sample_rate=8000, n_mfcc=13, n_fft=400, hop_length=160, n_mels=40):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # 预计算Mel滤波器组
        self.mel_filters = self._create_mel_filterbank()

        # 预计算DCT矩阵
        self.dct_matrix = self._create_dct_matrix()

    def _hz_to_mel(self, hz):
        """将频率从Hz转换为Mel刻度"""
        return 2595 * np.log10(1 + hz / 700.0)

    def _mel_to_hz(self, mel):
        """将频率从Mel刻度转换为Hz"""
        return 700 * (10 ** (mel / 2595.0) - 1)

    def _create_mel_filterbank(self):
        """创建Mel滤波器组"""
        # 计算频率范围
        low_freq = 0
        high_freq = self.sample_rate / 2
        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)

        # 在Mel刻度上均匀分布点
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # 转换为FFT bin索引
        fft_bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # 创建滤波器组
        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))

        for i in range(1, self.n_mels + 1):
            left = fft_bins[i - 1]
            center = fft_bins[i]
            right = fft_bins[i + 1]

            # 上升斜坡
            if left != center:
                filters[i - 1, left:center] = np.linspace(0, 1, center - left)

            # 下降斜坡
            if center != right:
                filters[i - 1, center:right] = np.linspace(1, 0, right - center)

        return filters

    def _create_dct_matrix(self):
        """创建DCT(离散余弦变换)矩阵"""
        dct_matrix = np.zeros((self.n_mfcc, self.n_mels))

        for i in range(self.n_mfcc):
            for j in range(self.n_mels):
                dct_matrix[i, j] = np.cos((np.pi * i / self.n_mels) * (j + 0.5))

        # 归一化
        dct_matrix[0, :] *= 1.0 / np.sqrt(self.n_mels)
        dct_matrix[1:, :] *= np.sqrt(2.0 / self.n_mels)

        return dct_matrix

    def _compute_power_spectrum(self, frame):
        """计算单帧的功率谱"""
        # 应用汉宁窗
        window = get_window("hann", self.n_fft)
        windowed_frame = frame * window

        # 计算FFT
        fft_result = fft.fft(windowed_frame, n=self.n_fft)

        # 取前n_fft/2+1个点(对称性)
        magnitude = np.abs(fft_result[: self.n_fft // 2 + 1])

        # 计算功率谱
        power_spectrum = (1.0 / self.n_fft) * (magnitude**2)

        return power_spectrum

    def _compute_mel_spectrum(self, power_spectrum):
        """计算Mel频谱"""
        # 应用Mel滤波器组
        mel_spectrum = np.dot(self.mel_filters, power_spectrum)

        # 转换为dB单位
        mel_spectrum = 10 * np.log10(np.maximum(mel_spectrum, 1e-10))

        return mel_spectrum

    def _compute_mfcc(self, mel_spectrum):
        """计算MFCC系数"""
        # 应用DCT
        mfcc = np.dot(self.dct_matrix, mel_spectrum)

        return mfcc

    def load_dat_file(self, filepath):
        """加载.dat音频文件"""
        with open(filepath, "rb") as f:
            # 假设.dat文件是16位PCM格式
            data = f.read()
            # 将字节数据转换为numpy数组
            samples = np.array(struct.unpack("<" + "h" * (len(data) // 2), data))
            return samples.astype(np.float32) / 32768.0  # 归一化到[-1, 1]

    def compute_mfcc(self, signal, max_length=100):
        """计算整个信号的MFCC特征"""
        # 确保信号是1D numpy数组
        if isinstance(signal, str):
            signal = self.load_dat_file(signal)
        elif isinstance(signal, list):
            signal = np.array(signal)

        # 分帧处理
        num_frames = 1 + (len(signal) - self.n_fft) // self.hop_length
        mfcc_features = []

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.n_fft

            if end > len(signal):
                frame = np.pad(signal[start:], (0, end - len(signal)))
            else:
                frame = signal[start:end]

            # 计算功率谱
            power_spectrum = self._compute_power_spectrum(frame)

            # 计算Mel频谱
            mel_spectrum = self._compute_mel_spectrum(power_spectrum)

            # 计算MFCC
            mfcc = self._compute_mfcc(mel_spectrum)

            mfcc_features.append(mfcc)

        # 转换为(n_mfcc, time)格式
        mfcc_features = np.array(mfcc_features).T

        # 特征长度标准化
        if mfcc_features.shape[1] > max_length:
            mfcc_features = mfcc_features[:, :max_length]
        else:
            pad_size = max_length - mfcc_features.shape[1]
            mfcc_features = np.pad(mfcc_features, ((0, 0), (0, pad_size)), mode="constant")

        return mfcc_features


class SpeechDataset(Dataset):
    def __init__(self, root_dir, vocab, mfcc_extractor, max_length=100):
        """
        Args:
            root_dir (str): 数据根目录
            vocab (dict): 词汇表 {序号: 单词}
            mfcc_extractor (ManualMFCC): MFCC提取器
            max_length (int): 最大MFCC序列长度
        """
        self.root_dir = root_dir
        self.vocab = vocab
        self.mfcc_extractor: ManualMFCC = mfcc_extractor
        self.max_length = max_length

        # 在初始化时预处理所有数据
        self.data = self._preprocess_all_data()

    def _build_file_list(self):
        """构建文件列表和对应的标签"""
        files = []
        for word_idx in self.vocab.keys():
            for repeat in range(1, 21):
                for id in ["21300240018", "21307130050", "21307130052", "21307130121", "21307130150", "22307130379"]:
                    filename = f"Data/{word_idx}/{id}_{word_idx}_{repeat:02d}.dat"
                    if os.path.exists(os.path.join(self.root_dir, filename)):
                        files.append((filename, int(word_idx)))

                    filename = f"Data/{word_idx}/{id}-{word_idx}-{repeat:02d}.dat"
                    if os.path.exists(os.path.join(self.root_dir, filename)):
                        files.append((filename, int(word_idx)))
        print(f"训练集: {len(files)} 个音频文件")
        return files

    def _preprocess_all_data(self):
        """预处理所有数据并存储在内存中"""
        file_list = self._build_file_list()
        data = []

        for filename, label in file_list:
            filepath = os.path.join(self.root_dir, filename)

            # 提取MFCC特征
            mfcc = self.mfcc_extractor.compute_mfcc(filepath)

            # 转换为torch张量并存储
            mfcc_tensor = torch.FloatTensor(mfcc)
            label_tensor = torch.LongTensor([label])

            data.append((mfcc_tensor, label_tensor))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接从内存中返回预处理好的数据
        return self.data[idx]


class SpeechRecognizer(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=20):
        super(SpeechRecognizer, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, time, features)
        x, _ = self.lstm(x)
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x
