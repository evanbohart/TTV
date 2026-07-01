import math
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
            self,
            d_model: int,
            vocab_size: int
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.e = nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, x):
        #x: (batch, seq_len)
        return self.e(x)

class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model: int,
            seq_len: int,
            dropout: float
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)

        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10*1000) / d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        #x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])

class LayerNorm(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        eps = 1e-6

        return self.gamma * (x - mean) / (std + eps) + self.beta

class FeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout: float
    ):
        super().__init__()

        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(torch.relu(self.lin1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        h: int,
        d_k: int,
        d_v: int,
        dropout: float
    ):
        assert d_model % h == 0

        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.w_Q = nn.Linear(d_model, h * d_k)
        self.w_K = nn.Linear(d_model, h * d_k)
        self.w_V = nn.Linear(d_model, h * d_v)
        self.w_O = nn.Linear(h * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]
        k_T = k.transpose(-2, -1) # (batch, h, d_k, seq_len)
        a = (q @ k_T) / math.sqrt(d_k) # (batch, h, seq_len, seq_len)

        if mask is not None:
            a.masked_fill_(mask, float('-inf'))

        a = dropout(a.softmax(-1))

        return (a @ v), a # (batch, h, seq_len, d_v), (batch, h, seq_len, seq_len)

    def forward(self, q, k, v, mask):
        Q = self.w_Q(q) # (batch, seq_len, h * d_k)
        K = self.w_K(k) # (batch, seq_len, h * d_k)
        V = self.w_V(v) # (batch, seq_len, h * d_k)

        Q = Q.view(
            Q.shape[0], Q.shape[1], self.h, self.d_k
        ).transpose(1, 2) # (batch, h, seq_len, d_k)

        K = K.view(
            K.shape[0], K.shape[1], self.h, self.d_k
        ).transpose(1, 2) # (batch, h, seq_len, d_k)

        V = V.view(
            V.shape[0], V.shape[1], self.h, self.d_v
        ).transpose(1, 2) # (batch, h, seq_len, d_v)

        x, self.a = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k
        ) # (batch, seq_len, h * d_k)

        return self.w_O(x) # (batch, seq_len, d_model)

class ResidualConnection(nn.Module):
    def __init__(
        self,
        dropout: float
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(x))

class EncoderBlock(nn.Module):
    def __init__(
        self,
        sa: MultiHeadAttention,
        ff: FeedForward,
        dropout: float
    ):
        super().__init__()

        self.sa = sa
        self.ff = ff
        self.res_connections = nn.ModuleList(
            [ResidualConnection(dropout) for i in range(2)]
        )

    def forward(self,
                x,
                src_padding_mask
    ):
        # self attention
        x = self.res_connections[0](
            x,
            lambda x: self.sa(
                x, x, x, src_padding_mask
            )
        )
        # feed forward
        x = self.res_connections[1](x, self.ff)

        return x

class DecoderBlock(nn.Module):
    def __init__(
        self,
        sa: MultiHeadAttention,
        ca: MultiHeadAttention,
        ff: FeedForward,
        dropout: float
    ):
        super().__init__()

        self.sa = sa
        self.ca = ca
        self.ff = ff
        self.res_connections = nn.ModuleList(
            [ResidualConnection(dropout) for i in range(3)]
        )

    def forward(
        self,
        x,
        encoder_output,
        src_padding_mask,
        tgt_padding_mask,
        tgt_causal_mask
    ):
        x = self.res_connections[0](
            x,
            lambda x: self.sa(
                x, x, x, tgt_padding_mask | tgt_causal_mask
            )
        )
        x = self.res_connections[1](
            x,
            lambda x: self.ca(
                x, encoder_output, encoder_output, src_padding_mask
            )
        )
        x = self.res_connections[2](x, self.ff)

        return x

class Encoder(nn.Module):
    def __init__(
        self,
        blocks: nn.ModuleList
    ):
        super().__init__()

        self.blocks = blocks
        self.norm = LayerNorm()

    def forward(self, x, src_padding_mask):
        for block in self.blocks:
            x = block(x, src_padding_mask)

        return self.norm(x)

class Decoder(nn.Module):
    def __init__(
        self,
        blocks: nn.ModuleList
    ):
        super().__init__()

        self.blocks = blocks
        self.norm = LayerNorm()

    def forward(
            self,
            x,
            encoder_output,
            src_padding_mask,
            tgt_padding_mask,
            tgt_causal_mask
    ):
        #x: (batch, seq_len, d_model)
        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                src_padding_mask,
                tgt_padding_mask,
                tgt_causal_mask
            )

        return self.norm(x)

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
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

        self.encoder = Encoder(
            nn.ModuleList([EncoderBlock(
                MultiHeadAttention(
                    d_model=d_model,
                    seq_len=encoder_seq_len,
                    h=h,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout
                ),
                FeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout
                ),
                dropout=dropout
            ) for i in range(n)])
        )
        self.decoder = Decoder(
            nn.ModuleList([DecoderBlock(
                MultiHeadAttention(
                    d_model=d_model,
                    seq_len=decoder_seq_len,
                    h=h,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout
                ),
                MultiHeadAttention(
                    d_model=d_model,
                    seq_len=decoder_seq_len,
                    h=h,
                    d_k=d_k,
                    d_v=d_v,
                    dropout=dropout
                ),
                FeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout
                ),
                dropout=dropout
            ) for i in range(n)])
        )

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        encoder_x,
        decoder_x,
        src_padding_mask,
        tgt_padding_mask,
        tgt_causal_mask
    ):
        src_padding_mask = src_padding_mask[:, None, None]
        tgt_padding_mask = tgt_padding_mask[:, None, None]
        tgt_causal_mask = tgt_causal_mask[None, None]

        # x: (batch, seq_len, d_model)
        encoder_x = self.encoder(
            encoder_x, src_padding_mask
        )

        decoder_x = self.decoder(
            decoder_x,
            encoder_x,
            src_padding_mask,
            tgt_padding_mask,
            tgt_causal_mask
        )

        # x -> (batch, seq_len, vocab_size)
        return self.linear(decoder_x)
