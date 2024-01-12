import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Ethan import Timer, Accumulator
from MyRNNdata import Vocab, truncate_pad
from Myplot import Animator


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.to(device)

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class Seq2SeqEncoder(Encoder):
    def __init__(
        self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    # X->(batch_size, num_steps, vocab_size)
    def forward(self, X, *args):
        # X->(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # X->(num_steps, batch_size, embed_size)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        # output->(num_steps,  batch_size, num_hiddens)
        # state-->(num_layers, batch_size, num_hiddens)
        return output, state


class Seq2SeqDecoder(Decoder):
    def __init__(
        self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # enc_outputs->(output, state)
        # return state
        return enc_outputs[1]

    # X------>(batch_size, num_steps)
    # state-->(num_layers, batch_size, num_hiddens)
    def forward(self, X, state):
        # X->(num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # state[-1]-->(batch_size, num_hiddens)
        # context---->(num_steps, batch_size, num_hiddens)
        context = state[-1].repeat(X.shape[0], 1, 1)
        # X_and_context->(num_steps, batch_size, embed_size+num_hiddens)
        X_and_context = torch.cat((X, context), 2)
        # output->(num_steps, batch_size, num_hiddens)
        # state-->(num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X_and_context, state)
        # output->(batch_size, num_steps, vocab_size)
        output = self.dense(output).permute(1, 0, 2)
        return output, state


def sequence_mask(X, valid_len, value=0):
    # X->[[],[],[],[]]
    maxlen = X.size(1)
    # torch.arange(maxlen)[None, :]->[[0,1,2,3]]
    # valid_len[:, None]->[[2],[2],[3],[3]]
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # pred------>(batch_size, num_steps, vocab_size)
    # label----->(batch_size, num_steps)
    # valid_len->(batch_size, 1)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    loss_fn = MaskedSoftmaxCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    animator = Animator(xlabel="epoch", ylabel="loss", xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos_index = torch.tensor(
                [tgt_vocab["<bos>"]] * Y.shape[0], device=device
            ).reshape(-1, 1)
            dec_input = torch.cat([bos_index, Y[:, :-1]], 1)
            pred, _ = net(X, dec_input, X_valid_len)
            loss = loss_fn(pred, Y, Y_valid_len)
            loss.sum().backward()
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(
        f"loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} "
        f"tokens/sec on {str(device)}"
    )


def pred_seq2seq(
    net,
    src_sentence: str,
    src_vocab,
    tgt_vocab,
    num_steps,
    device="cuda",
    save_attention_weights=False,
):
    net.eval()
    src_tokens: list[int] = src_vocab[src_sentence.lower().split(" ")] + [
        src_vocab["<eos>"]
    ]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens: list[int] = truncate_pad(src_tokens, num_steps, src_vocab["<pad>"])
    # enc_X->(batch_size=1, num_steps)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )
    # enc_outputs->(output, state)
    enc_outputs: tuple = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # dec_X->(batch_size=1, len=1)
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device), dim=0
    )

    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # Y->(batch_size=1, len=1, vocab_size)
        Y, dec_state = net.decoder(dec_X, dec_state)
        # dec_X->(batch_size=1, len=1)
        dec_X = Y.argmax(dim=2)
        # pred->(1)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)
    return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

import math
import collections

def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(" "), label_seq.split(" ")
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i : i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[" ".join(pred_tokens[i : i + n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i : i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# ------------------------------------------


class RNNModle(nn.Module):
    def __init__(self, rnn_layer: nn.RNN, vocab_size: int, device="cuda"):
        super().__init__()
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
        self.to(device)

    # input->(batch_size, num_steps)
    # X->(num_steps, batch_size, vocab_size)
    # state->(1, batch_size, num_hiddens)
    # Y->(num_steps, batch_size, num_hiddens)
    # output->(num_steps*batch_size, vocab_size)
    def forward(self, input: torch.Tensor, state: torch.Tensor):
        X = F.one_hot(input.T, self.vocab_size).to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, batch_size=1, device="cuda"):
        if isinstance(self.rnn, nn.LSTM):
            return (
                torch.zeros(
                    size=(
                        self.rnn.num_layers * self.num_directions,
                        batch_size,
                        self.num_hiddens,
                    ),
                    device=device,
                ),
                torch.zeros(
                    size=(
                        self.rnn.num_layers * self.num_directions,
                        batch_size,
                        self.num_hiddens,
                    ),
                    device=device,
                ),
            )
        return torch.zeros(
            size=(
                self.rnn.num_layers * self.num_directions,
                batch_size,
                self.num_hiddens,
            ),
            device="cuda",
        )


# get_input(->(batch_size=1, num_steps=1)
# y->(num_steps*batch_size=1, vocab_size)
def pred_RNN(model: RNNModle, prefix: str, num_preds: int, vocab: Vocab):
    prefix: list[str] = prefix.split()
    state = model.begin_state(batch_size=1)
    outputs: list[int] = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor(outputs[-1], device="cuda").reshape(1, 1)
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = model(get_input(), state)
        y = int(y.argmax(dim=1).reshape(1))
        outputs.append(y)
    return " ".join([vocab.to_tokens(index) for index in outputs])


def train_RNN(
    model: RNNModle,
    train_iter: DataLoader,
    vocab: Vocab,
    epochs: int,
    lr: float,
    device="cuda",
):
    animator = Animator(
        xlabel="epoch", ylabel="perplexity", legend=["train"], xlim=[10, epochs]
    )
    pred_text = lambda prefix: pred_RNN(model, prefix, 50, vocab)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for t in range(epochs):
        ppl, speed = train_epoch(
            model, train_iter, train_iter.batch_size, loss_fn, optimizer, device
        )
        if (t + 1) % 10 == 0:
            print(pred_text("time traveller"))
            animator.add(t + 1, ppl)
    print(f"困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}")
    print(pred_text("time traveller"))
    print(pred_text("traveller"))


def train_epoch(
    model: RNNModle,
    train_iter: DataLoader,
    batch_size: int,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device="cuda",
):
    timer = Timer()
    sum_loss, sum_tokens = 0.0, 0
    state = model.begin_state(batch_size, device)
    # X->(batch_size, num_steps)
    # Y->(batch_size, num_steps)
    for X, Y in train_iter:
        if isinstance(state, tuple):
            for s in state:
                s.detach_()
        else:
            state.detach_()
        # Y->(num_steps*batch_size)
        Y = Y.T.reshape(-1)
        X, Y = X.to(device), Y.to(device)
        # pred->(num_steps*batch_size, vocab_size)
        pred, state = model(X, state)
        loss = loss_fn(pred, Y.long()).mean()

        loss.backward()
        grad_clipping(model, 1)
        optimizer.step()
        optimizer.zero_grad()

        sum_loss += loss * Y.numel()
        sum_tokens += Y.numel()
    ppl = torch.exp(sum_loss / sum_tokens).item()
    speed = sum_tokens / timer.stop()
    return ppl, speed


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
