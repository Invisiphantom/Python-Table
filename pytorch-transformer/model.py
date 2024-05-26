import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    # (batch, seq_len) --> (batch, seq_len, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)


# ----------------------------------------------------------------------------------------
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod  # (batch_size, h, seq_len, d_model)
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k)@(batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # Apply softmax
        attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len)@(batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        return attention_scores @ value

    def forward(self, q, k, v, mask):
        q = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        k = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        v = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

        # (batch, h, seq_len, d_k)
        x = MultiHeadAttentionBlock.attention(q, k, v, mask, self.dropout)
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# ----------------------------------------------------------------------------------------


class LayerNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class AddNorm(nn.Module):

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ----------------------------------------------------------------------------------------


class EncoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        addnorm1: AddNorm,
        addnorm2: AddNorm,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.addnorm1 = addnorm1
        self.addnorm2 = addnorm2

    def forward(self, x, src_mask):
        x = self.addnorm1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.addnorm2(x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, encode_blocks: nn.ModuleList, encode_norm) -> None:
        super().__init__()
        self.encode_blocks = encode_blocks
        self.encode_norm = encode_norm

    def forward(self, x, mask):
        for enc_block in self.encode_blocks:
            x = enc_block(x, mask)
        return self.encode_norm(x)


# ----------------------------------------------------------------------------------------


class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        addnorm1: AddNorm,
        addnorm2: AddNorm,
        addnorm3: AddNorm,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.addnorm1 = addnorm1
        self.addnorm2 = addnorm2
        self.addnrom3 = addnorm3

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.addnorm1(x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.addnorm2(x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.addnrom3(x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, decode_blocks: nn.ModuleList, decode_norm: LayerNorm) -> None:
        super().__init__()
        self.decode_blocks = decode_blocks
        self.decode_norm = decode_norm

    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        for dec_block in self.decode_blocks:
            tgt = dec_block(tgt, encoder_output, src_mask, tgt_mask)
        return self.decode_norm(tgt)


# ----------------------------------------------------------------------------------------


class ProjLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)


# ----------------------------------------------------------------------------------------


class Transformer(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        proj_layer: ProjLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    # (batch, seq_len, d_model)
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    # (batch, seq_len, d_model)
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
    def project(self, x):
        return self.proj_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = nn.ModuleList()
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # W_q W_k W_v W_o
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # w1 b1 w2 b2
        addnorm1 = AddNorm(d_model, dropout)  # gamma beta
        addnorm2 = AddNorm(d_model, dropout)  # gamma beta
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, addnorm1, addnorm2)
        encoder_blocks.append(encoder_block)

    decoder_blocks = nn.ModuleList()
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # W_q W_k W_v W_o
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # W_q W_k W_v W_o
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # w1 b1 w2 b2
        addnrom1 = AddNorm(d_model, dropout)  # gamma beta
        addnorm2 = AddNorm(d_model, dropout)  # gamma beta
        addnorm3 = AddNorm(d_model, dropout)  # gamma beta
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, addnrom1, addnorm2, addnorm3)
        decoder_blocks.append(decoder_block)

    encoder_norm = LayerNorm(d_model)  # gamma beta
    decoder_norm = LayerNorm(d_model)  # gamma beta
    encoder = Encoder(encoder_blocks, encoder_norm)
    decoder = Decoder(decoder_blocks, decoder_norm)
    projection_layer = ProjLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
