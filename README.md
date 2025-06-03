

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source .bashrc

sudo chmod -R 777 /opt/
pip install numpy pandas matplotlib jupyter import-ipynb scikit-learn
pip install torch torchvision torchaudio torchmetrics tensorboard torch-tb-profiler
pip install transformers datasets tokenizers
```


