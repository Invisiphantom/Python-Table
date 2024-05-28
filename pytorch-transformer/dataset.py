import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class BilingualDataset(Dataset):

    def __init__(
        self, ds_raw, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang: str, tgt_lang: str, seq_len: int
    ):
        super().__init__()

        # id, translation{en, it}
        self.ds_raw = ds_raw
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)  # [SOS]
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)  # [EOS]
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)  # [PAD]

    def __len__(self):
        return len(self.ds_raw)

    def __getitem__(self, idx):
        src_target_pair = self.ds_raw[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # 将文本转换为token
        enc_input_tokens = torch.tensor(self.tokenizer_src.encode(src_text).ids, dtype=torch.int64)
        dec_input_tokens = torch.tensor(self.tokenizer_tgt.encode(tgt_text).ids, dtype=torch.int64)

        # 编码器token的空白数 (需要减去<s>和</s>)
        # 解码器token的空白数 (需要减去<s>或</s>)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # 确保预设的seq_len足够长
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("seq_len太小了")

        encoder_input = torch.cat(
            [
                self.sos_token,
                enc_input_tokens,
                self.eos_token,
                self.pad_token.repeat(enc_num_padding_tokens),
            ]
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                dec_input_tokens,
                self.pad_token.repeat(dec_num_padding_tokens),
            ]
        )
        label = torch.cat(
            [
                dec_input_tokens,
                self.eos_token,
                self.pad_token.repeat(dec_num_padding_tokens),
            ]
        )

        # 确保所有的文本的长度都为seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        sequence_mask = torch.ones(1, self.seq_len, self.seq_len, dtype=torch.bool).tril() # 位置掩码
        encoder_mask = (encoder_input != self.pad_token).repeat(self.seq_len, 1).unsqueeze(0)
        decoder_mask = (decoder_input != self.pad_token).repeat(self.seq_len, 1).unsqueeze(0) & sequence_mask

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": encoder_mask.to(torch.int64),  # (1, 1, seq_len)
            "decoder_mask": decoder_mask.to(torch.int64),  # (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }