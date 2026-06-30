import torch
import torch.nn as nn

from .transformer import Embedding, PositionalEncoding, Transformer

class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        encoder_x_dim: int,
        encoder_seq_len: int,
        decoder_seq_len: int,
        h: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        n: int,
        dropout: float
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.encoder_proj = nn.Linear(encoder_x_dim, d_model)
        self.decoder_embed = Embedding(
            vocab_size=vocab_size,
            d_model=d_model
        )

        self.pe = PositionalEncoding(
            d_model=d_model,
            seq_len=encoder_seq_len,
            dropout=dropout
        )

        self.transformer = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_seq_len=encoder_seq_len,
            decoder_seq_len=decoder_seq_len,
            h=h,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            n=n,
            dropout=dropout
        )

    def forward(
        self,
        encoder_x,
        decoder_x,
        src_padding_mask,
        tgt_padding_mask,
        tgt_causal_mask
    ):
        encoder_x = self.encoder_proj(encoder_x)
        encoder_x = self.pe(encoder_x)

        decoder_x = self.decoder_embed(decoder_x)
        decoder_x = self.pe(decoder_x)

        return self.transformer(
            encoder_x,
            decoder_x,
            src_padding_mask,
            tgt_padding_mask,
            tgt_causal_mask
        )
