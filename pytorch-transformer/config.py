from pathlib import Path

def get_config():
    return {
        "dataset": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        
        "tokenizer": "tokenizer_{0}.json",
        "tokenizer_dir": "Tokenizer",
        
        "preload": "latest",
        "model_dir": "model",
        "model_name": "tmodel_",
        "tb_dir": "runs/tmodel",
        
        "lr": 1e-4,
        "num_epochs": 1,
        "batch_size": 8,
        "seq_len": 350,
        "d_model": 512,
    }

def get_model_path(config, epoch: str):
    Path(f"{config['dataset']}_{config['model_dir']}").mkdir(parents=True, exist_ok=True)
    model_dir = f"{config['dataset']}_{config['model_dir']}"
    model_filename = f"{config['model_name']}{epoch}.pt"
    return str(Path('.') / model_dir / model_filename)

def latest_weights_file_path(config):
    Path(f"{config['dataset']}_{config['model_dir']}").mkdir(parents=True, exist_ok=True)
    model_dir = f"{config['dataset']}_{config['model_dir']}"
    model_filename = f"{config['model_name']}*"
    weights_files = list(Path(model_dir).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
