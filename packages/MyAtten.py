import math
import torch
import torch.nn as nn
from MyRNNdata import load_data_nmt
from MyRNN import Encoder, Decoder, EncoderDecoder, Seq2SeqEncoder, train_seq2seq


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


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, query_size, key_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries-->(batch_size, query_num, query_size) (2,1,20)
        # key------>(batch_size,   key_num,   key_size) (2,10,2)
        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries-->(batch_size, query_num,       1, num_hiddens)
        # key------>(batch_size,         1, key_num, num_hiddens)
        # features->(batch_size, query_num, key_num, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # scores------------->(batch_size, query_num, key_num)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # weights->(batch_size, query_num,   key_num)
        # values-->(batch_size,   key_num, value_num)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries---->(batch_size, query_num, d)
    # keys------->(batch_size, key_num, d)
    # values----->(batch_size, key_num, value_num)
    # valid_lens->(batch_size, )or(batch_size, query_num)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # mat1---->(batch_size, query_num,       d)
        # mat2---->(batch_size,         d, key_num)
        # scores-->(batch_size, query_num, key_num)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # return->(batch_size, query_num, value_num)
        return torch.bmm(self.dropout(self.attention_weights), values)


# -----------------------------------


class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(
        self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout
        )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    # output->(num_steps,  batch_size, num_hiddens)
    # state-->(num_layers, batch_size, num_hiddens)
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    # X------>(batch_size, num_steps)
    # state-->(enc_output, hidden_state, enc_valid_lens)
    def forward(self, X, state):
        # enc_output---->(batch_size,  num_steps, num_hiddens).
        # hidden_state-->(num_layers, batch_size, num_hiddens)
        enc_output, hidden_state, enc_valid_lens = state
        # X->(num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        # x->(batch_size, num_steps)
        for x in X:
            # query->(batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context->(batch_size, 1, num_hiddens)
            context = self.attention(
                queries=query,
                keys=enc_output,
                values=enc_output,
                valid_lens=enc_valid_lens,
            )
            # x->(batch_size, 1, embed_size+num_hiddens)
            x = torch.cat((torch.unsqueeze(x, dim=1), context), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            # out----->(        1, batch_size. num_hiddens)
            # outputs->(num_steps, batch_size. num_hiddens)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # outputs->(num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_output, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


def Seq2SeqAttention_train(
    embed_size=32,
    num_hiddens=32,
    num_layers=2,
    dropout=0.1,
    batch_size=64,
    num_steps=10,
    lr=0.005,
    num_epochs=250,
    device="cuda",
):
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# ------------------------------------


def transpose_qkv(X, num_heads):
    # X->(batch_size, query_num or key_num, num_hiddens)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # X->(batch_size, query_num or key_num, num_heads, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # X->(batch_size,  num_heads, query_num or key_num, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])
    # return->(batch_size * num_heads, query_num or key_num, num_hiddens/num_heads)


def transpose_output(output, num_heads):
    # output->(batch_size * num_heads, query_num, num_hiddens/num_heads)
    output = output.reshape(-1, num_heads, output.shape[1], output.shape[2])
    # output->(batch_size, num_heads, query_num, num_hiddens/num_heads)
    output = output.permute(0, 2, 1, 3)
    # output->(batch_size, query_num, num_heads, num_hiddens/num_heads)
    return output.reshape(output.shape[0], output.shape[1], -1)
    # return->(batch_size, query_num, num_hiddens)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        num_heads,
        dropout,
        bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    # queries---->(batch_size, query_num, d)
    # keys------->(batch_size,   key_num, d)
    # values----->(batch_size,   key_num, value_num)
    # valid_lens->(batch_size, )or(batch_size, query_num)
    def forward(self, queries, keys, values, valid_lens=None):
        # W_q(queries)---->(batch_size, query_num, num_hiddens)
        # W_k(keys)------->(batch_size,   key_num, num_hiddens)
        # W_v(values)----->(batch_size,   key_num, num_hiddens)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # queries---->(batch_size * num_heads, query_num, num_hiddens/num_heads)
        # keys------->(batch_size * num_heads,   key_num, num_hiddens/num_heads)
        # values----->(batch_size * num_heads,   key_num, num_hiddens/num_heads)

        if valid_lens is not None:
            # valid_lens->(batch_size * num_heads, )or(batch_size * num_heads, query_num)
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # output->(batch_size * num_heads, query_num, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat->(batch_size, quety_num, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# ---------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    # X->(1, num_steps, num_hiddens)
    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


# ---------------------------------


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, Skip_X, Y):
        return self.ln(Skip_X + self.dropout(Y))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input: int,
        ffn_num_hiddens: int,
        num_heads: int,
        dropout: float,
        use_bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(
            X, self.attention(queries=X, keys=X, values=X, valid_lens=valid_lens)
        )
        return self.addnorm2(X, self.ffn(Y))


class TransformerEncoder(Encoder):
    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    use_bias,
                ),
            )

    # X->(batch_size, num_steps, vocab_size)
    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        i,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    # X->(batch_size, num_steps, num_hiddens)
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens->(batch_size, num_steps) mask
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(
                batch_size, 1
            )
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    i,
                ),
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


def train_Transformer():
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, "cuda"
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(
        len(src_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )
    decoder = TransformerDecoder(
        len(tgt_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
