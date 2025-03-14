

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source .bashrc

pip install torch torchvision torchaudio
pip install numpy pandas matplotlib jupyter
pip install scikit-learn transformers datasets tokenizers tqdm tensorboard torchmetrics
```


```py
# 最高验证准确率: 96.90%
net_info = [
    {"in_features": 784, "out_features": 256, "activation": Sigmoid},
    {"in_features": 256, "out_features": 64, "activation": Sigmoid},
    {"in_features": 64, "out_features": 16, "activation": Sigmoid},
    {"in_features": 16, "out_features": 10, "activation": Id},
]
```