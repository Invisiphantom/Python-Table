import os
import re
import random
import zipfile
import tarfile
import hashlib
import requests
import collections
import torch
from torch.utils.data import DataLoader
from Ethan import MyDataset

DATA_HUB = dict()
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATA_HUB["time_machine"] = (
    DATA_URL + "TimeMachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)
DATA_HUB["fra-eng"] = (
    DATA_URL + "fra-eng.zip",
    "94646ad1522d915e7b0f9296181140edcf86a4f5",
)


def download_extract(name: str, folder=None):
    """Download and extract a zip/tar file.

    Defined in :numref:`sec_utils`"""
    fname = URL_Data(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "Only zip/tar files can be extracted."
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


# raw_text
def read_data_nmt():
    data_dir = download_extract("fra-eng")
    with open(os.path.join(data_dir, "fra.txt"), "r", encoding="utf-8") as f:
        return f.read()


# raw_text -> text
def preprocess_nmt(text: str):
    def no_space(char, prev_char):
        return char in set(",.!?") and prev_char != " "

    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    out = [
        " " + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return "".join(out)


# text -> source, target
def tokenize_nmt(text: str, num_examples=None):
    source, target = [], []
    # line -> ["go .  va !", "hi .  salut !", ...]
    for i, line in enumerate(text.split("\n")):
        if num_examples and i > num_examples:
            break
        # parts -> [["go ."],["va !"]]
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    # source -> [['go', '.'], ...], target ->  [['va', '!'], ...]
    return source, target


def truncate_pad(line: list[int], num_steps, padding_token):
    # line -> [47, 1]
    if len(line) > num_steps:
        return line[:num_steps]
    # return -> [47, 4, 1, 1, 1, 1, 1, 1, 1, 1]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines: list[str], vocab, num_steps):
    lines: list[int] = [vocab[l] for l in lines]
    lines: list[int] = [l + [vocab["<eos>"]] for l in lines]
    # vocab["<eos>"] ------> 3
    # array -> [36, 10,  4,  3,  1,  1,  1,  1]
    array = torch.tensor([truncate_pad(l, num_steps, vocab["<pad>"]) for l in lines])
    valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays: tuple, batch_size: int, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_nmt(batch_size: int, num_steps: int, num_examples=600):
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    # source -> [['go', '.'], ...], target ->  [['va', '!'], ...]
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    # src_array -> [[9, 4, 3, 1, 1, 1, 1, 1], [113, 4, 3, 1, 1, 1, 1, 1], ...]
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab



# -------------------------------------------------------------


# TimeMachine.txt
def URL_Data(url, folder="../data", sha1_hash=None):
    if not url.startswith("http"):
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    name = os.path.join(folder, url.split("/")[-1])
    if os.path.exists(name) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(name, "rb") as f:
            while True:
                data = f.read(1024 * 1024)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return name
    print(f"Downloading {name} from {url}...")
    r = requests.get(url, stream=True, verify=True)
    with open(name, "wb") as f:
        f.write(r.content)
    return name


# lines
def TimeMachine_Lines():
    with open(URL_Data("time_machine"), "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


# tokens
def tokenize(lines):
    return [line.split() for line in lines]


def tokenFlatten(tokens):
    if tokens and isinstance(tokens[0], list):
        return [token for line in tokens for token in line]
    return tokens


# vocab
class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = list(
            ["<unk>"]
            + reserved_tokens
            + [token for token, freq in self.token_freqs if freq >= min_freq]
        )
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens: str | list[str]):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices: int | list[int]):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return self.token_to_idx["<unk>"]


# dataloader shape=(num_steps, batch_size, token_types)
def randomSeq_DataLoader(corpus: list, batch_size: int, num_steps: int):
    corpus = corpus[random.randint(0, num_steps - 1) :]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos : pos + num_steps]

    X = torch.tensor([data(i) for i in initial_indices])
    Y = torch.tensor([data(i + 1) for i in initial_indices])

    randomSeq_dataset = MyDataset(X, Y)
    dataloader = DataLoader(randomSeq_dataset, batch_size)
    return dataloader


def sequentialSeq_DataLoader(corpus: list, batch_size: int, num_steps: int):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset : offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)

    num_batches = Xs.shape[1] // num_steps

    X, Y = [], []
    for i in range(0, num_steps * num_batches, num_steps):
        X += Xs[:, i : i + num_steps].reshape(-1).tolist()
        Y += Ys[:, i : i + num_steps].reshape(-1).tolist()

    X = torch.tensor(X).reshape(-1, num_steps)
    Y = torch.tensor(Y).reshape(-1, num_steps)

    sequentialSeq_dataset = MyDataset(X, Y)
    dataloader = DataLoader(sequentialSeq_dataset, batch_size)
    return dataloader


def TimeMachine_Vocab_DL(
    batch_size: int, num_steps: int, max_tokens=10000, random=False
):
    lines = TimeMachine_Lines()
    tokens = tokenize(lines)
    Flat_tokens: list[str] = tokenFlatten(tokens)

    vocab = Vocab(Flat_tokens)
    corpus: list[int] = [vocab[token] for line in tokens for token in line]
    corpus: list[int] = corpus[:max_tokens]
    if random == False:
        dataloader = sequentialSeq_DataLoader(corpus, batch_size, num_steps)
    else:
        dataloader = randomSeq_DataLoader(corpus, batch_size, num_steps)
    return vocab, dataloader
