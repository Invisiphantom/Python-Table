import math
import torch
import torch.nn as nn


# (batch, seq_len) --> (batch, seq_len, d_model)
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# (batch, seq_len, d_model)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        tensor_2i = torch.arange(0, d_model, 2, dtype=torch.float)  # (d_model / 2)
        div_term = torch.exp(tensor_2i * (-math.log(10000.0) / d_model))  # (d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (-2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (-2i / d_model))
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------------


# (batch_size, h, seq_len, d_model)
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model不能被h整除!"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask, dropout: nn.Dropout = None):
        d_k = query.shape[-1]

        # (batch, h, qry_len, d_k)@(batch, h, d_k, seq_len) --> (batch, h, qry_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len)@(batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        return attention_scores @ value, attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
        q = self.w_q(q)  # (batch, qry_len, d_model) --> (batch, qry_len, d_model)
        k = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        v = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)
        x, _ = MultiHeadAttentionBlock.attention(q, k, v, mask, self.dropout)
        del q, k, v

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        return self.w_o(x)


# fmt:off
# (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)


# ----------------------------------------------------------------------------------------


# fmt:on
# (batch, seq_len, d_model)
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
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, src, src_mask):
        src = self.addnorm1(src, lambda x: self.self_attention_block(x, x, x, src_mask))
        src = self.addnorm2(src, self.feed_forward_block)
        return src


class Encoder(nn.Module):

    def __init__(self, encode_blocks: nn.ModuleList, encode_norm) -> None:
        super().__init__()
        self.encode_blocks = encode_blocks
        self.encode_norm = encode_norm

    def forward(self, src, src_mask):
        for encode_block in self.encode_blocks:
            src = encode_block(src=src, src_mask=src_mask)
        return self.encode_norm(src)


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

    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.addnorm1(tgt, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        tgt = self.addnorm2(tgt, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        tgt = self.addnrom3(tgt, self.feed_forward_block)
        return tgt


class Decoder(nn.Module):

    def __init__(self, decode_blocks: nn.ModuleList, decode_norm: LayerNorm) -> None:
        super().__init__()
        self.decode_blocks = decode_blocks
        self.decode_norm = decode_norm

    def forward(self, tgt, encoder_output, src_mask, tgt_mask):
        for dec_block in self.decode_blocks:
            tgt = dec_block(tgt=tgt, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.decode_norm(tgt)


# ----------------------------------------------------------------------------------------


class ProjLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return nn.functional.log_softmax(self.proj(x), dim=-1)


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
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.proj_layer = proj_layer

    # (batch, seq_len, d_model)
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src=src, src_mask=src_mask)

    # (batch, seq_len, d_model)
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt=tgt, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)

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
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            addnrom1,
            addnorm2,
            addnorm3,
        )
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
