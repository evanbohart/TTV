import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as transforms
import torchaudio.functional as functional

from model import Model
from tokenizer import tokenize, generate_tokens
from build_data import build_df

n_mels = 64
d_model = 512
encoder_seq_len = 512
decoder_seq_len = 64
h = 8
d_k = d_v = d_model // h
d_ff = 2048
n = 6
dropout = 0.1

df = build_df()

vocab = generate_tokens(df)
vocab_size = len(vocab)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    encoder_x = []
    decoder_x = []
    src_padding_mask = []
    tgt_padding_mask = []
    tgt_causal_mask = torch.triu(
        torch.ones(decoder_seq_len, decoder_seq_len, dtype=torch.bool),
        diagonal=1
    )
    targets = []

    batch_num = 0
    count = 0

    for i, sample in df.iterrows():
        path = sample['path']

        waveform, sr = torchaudio.load(path)
        waveform = functional.resample(waveform, sr, 8000)

        mel = transforms.MelSpectrogram(
            sample_rate=8000,
            n_fft=400,
            hop_length=256,
            n_mels=n_mels,
        )(waveform)

        mel = transforms.AmplitudeToDB()(mel)
        mel = mel.squeeze(0).transpose(0, 1)

        encoder_x_unpadded = (mel - mel.mean()) / (mel.std() + 1e-9)

        encoder_x_len = encoder_x_unpadded.shape[0]
        encoder_len_diff = encoder_seq_len - encoder_x_len

        tokens = tokenize(str(sample['transcript']))

        decoder_x_len = len(tokens) + 1
        decoder_len_diff = decoder_seq_len - decoder_x_len

        if encoder_len_diff >= 0 and decoder_len_diff >= 0:
            encoder_x.append(
                nn.functional.pad(encoder_x_unpadded, (0, 0, 0, encoder_len_diff))
            )

            src_padding_mask.append(
                torch.arange(encoder_seq_len) >= encoder_x_len
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

            print(targets)

            tgt_padding_mask.append(
                torch.arange(decoder_seq_len, device=device) >= decoder_x_len
            )

            count += 1

            if count == batches:
                encoder_x_batch = torch.stack(encoder_x).to(device)
                decoder_x_batch = torch.stack(decoder_x).to(device)
                src_padding_mask_batch = torch.stack(src_padding_mask).to(device)
                tgt_padding_mask_batch = torch.stack(tgt_padding_mask).to(device)
                tgt_causal_mask = tgt_causal_mask.to(device)
                targets_batch = torch.stack(targets).to(device)

                batch_num += 1

                for i in range(10000):
                    print('Feeding forward...')

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        logits = model(
                            encoder_x_batch,
                            decoder_x_batch,
                            src_padding_mask_batch,
                            tgt_padding_mask_batch,
                            tgt_causal_mask
                        )
                        loss = criterion(
                            logits.view(-1, vocab_size),
                            targets_batch.view(-1)
                        )

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()

                    print(f"Batch: {batch_num} | Loss: {loss.item():.4f}")

                encoder_x.clear()
                decoder_x.clear()
                src_padding_mask.clear()
                tgt_padding_mask.clear()
                targets.clear()

                count = 0

model = Model(
    encoder_x_dim=n_mels,
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

#nn.utils.clip_grad_norm_(model.parameters(), 1.0)
model.train()

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9,0.98), eps=1e-9)

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
    optimizer,
    scheduler,
    batches
)
