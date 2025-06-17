import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# 读取EDF文件
raw = mne.io.read_raw_edf('2.edf', preload=True)
raw.info['line_freq'] = 50  # 设置工频频率

# 读取睡眠标注
annotations = pd.read_csv('2.csv')
stages = annotations[annotations['Annotation'].isin(['W', 'N1', 'N2', 'N3', 'REM'])]

# 创建MNE标注对象
onset = stages['Onset'].values
duration = [30] * len(onset)  # 假设每个阶段持续30秒
description = stages['Annotation'].values
annot = mne.Annotations(onset, duration, description)
raw.set_annotations(annot)

# 选择感兴趣的通道
picks = ['Fp1', 'C3', 'O1', 'Cb1']
raw.pick(picks)

# 预处理
raw.filter(0.5, 45, fir_design='firwin')
raw.notch_filter(50)

# 绘制原始数据+标注
raw.plot(duration=60, scalings='auto', block=True)