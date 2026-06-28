import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import sys

from model import Model
from tokenizer import tokenize, generate_tokens
from build_data import build_df

def train(
    df,
    n_mels,
    encoder_seq_len,
    decoder_seq_len,
    model,
    criterion,
    optimizer,
    scheduler,
    batches
):
    encoder_x = []
    decoder_x = []
    src_padding_mask = []
    tgt_padding_mask = []
    tgt_casual_mask = []
    targets = []

    for i, sample in df.iterrows():
        path = sample['path']

        y, sr = librosa.load(path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        encoder_x_seq = torch.from_numpy(mel_db).float().transpose(0, 1)
        mean = encoder_x_seq.mean(0, keepdim=True)
        std = encoder_x_seq.std(0, keepdim=True)
        encoder_x_seq = (encoder_x_seq - mean) / (std + 1e-6)

        encoder_x_len = encoder_x_seq.shape[0]
        encoder_len_diff = encoder_seq_len - encoder_x_len

        tokens = tokenize(str(sample['transcript']))

        decoder_x_len = len(tokens) + 1
        decoder_len_diff = decoder_seq_len - decoder_x_len

        if encoder_len_diff >= 0 and decoder_len_diff >= 0:
            encoder_x.append(
                nn.functional.pad(encoder_x_seq, (0, 0, 0, encoder_len_diff))
            )
            src_padding_mask.append(
                torch.arange(encoder_seq_len) < encoder_x_len
            )

            decoder_x.append(
                torch.zeros(decoder_seq_len, dtype=torch.long)
            )
            decoder_x[-1][0] = vocab['<BOS>']

            targets.append(
                torch.full((decoder_seq_len,), -1, dtype=torch.long)
            )

            for j, token in enumerate(tokens):
                token_id = vocab[token]
                decoder_x[-1][j+1] = token_id
                targets[-1][j] = token_id

            targets[-1][decoder_x_len-1] = vocab['<EOS>']

            tgt_padding_mask.append(
                torch.arange(decoder_seq_len) < decoder_x_len
            )
            tgt_casual_mask.append(
                torch.tril(
                    torch.ones(decoder_seq_len, decoder_seq_len, dtype=torch.bool),
                    diagonal=1
                )
            )

            if (i+1) % batches == 0 or i == len(df) - 1:
                print('Feeding forward...')

                encoder_x_batch = torch.stack(encoder_x)
                decoder_x_batch = torch.stack(decoder_x)
                src_padding_mask_batch = torch.stack(src_padding_mask)
                tgt_padding_mask_batch = torch.stack(tgt_padding_mask)
                tgt_casual_mask_batch = torch.stack(tgt_casual_mask)
                targets_batch = torch.stack(targets)

                logits = model(
                    encoder_x_batch,
                    decoder_x_batch,
                    src_padding_mask_batch,
                    tgt_padding_mask_batch,
                    tgt_casual_mask_batch
                )
                loss = criterion(
                    logits.view(-1, vocab_size),
                    targets_batch.view(-1)
                )
                loss.backward()

                if optimizer:
                    optimizer.step()
                if scheduler:
                    scheduler.step()

                print(f"Loss: {loss.item():.4f}")

                encoder_x.clear()
                decoder_x.clear()
                src_padding_mask.clear()
                tgt_padding_mask.clear()
                tgt_casual_mask.clear()
                targets.clear()

n_mels = 128
d_model = 512
encoder_seq_len = 500
decoder_seq_len = 50
h = 8
d_k = d_v = d_model // h
d_ff = 2048
n = 6
dropout = 0.1

df = build_df()

vocab = generate_tokens(df)
vocab_size = len(vocab)

model = Model(
    encoder_x_dim = n_mels,
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

nn.utils.clip_grad_norm_(model.parameters(), 1.0)
model.train()

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.98), eps=1e-9)

warmup_steps = 4000

def lr_lambda(step):
    step = max(step, 1)

    return (d_model ** -0.5) * min(
        step ** -0.5,
        step * (warmup_steps ** -1.5)
    )

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

batches = 30

train(
    df,
    n_mels,
    encoder_seq_len,
    decoder_seq_len,
    model,
    criterion,
    None,
    None,
    batches
)
