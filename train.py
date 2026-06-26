from pathlib import Path
import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.optim as optim

from model import Model

dir = Path('LJSpeech-1.1')

df = pd.read_csv(dir.name+'/metadata.csv', sep='|')
transcriptions = df['Normalized_Transcription']

token_list = {}

def tokenize(text):
    tokens = []

    for word in text.split():
        token = "".join(
            char.lower() if char.isalnum() else ''
            for char in word
        )

        if token:
            tokens.append(token)

    return tokens

token_list["<BOS>"] = 0
token_list["<EOS>"] = 1

next_id = 2

for transcription in transcriptions:
    tokens = tokenize(str(transcription))

    for token in tokens:
        if token and token not in token_list:
            token_list[token] = next_id
            next_id += 1

def train(
    df,
    files,
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

    count = len(files)

    for i, file in enumerate(files):
        y, sr = librosa.load(file, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        encoder_x_seq = torch.from_numpy(mel_db).float().transpose(0, 1)
        mean = encoder_x_seq.mean(0, keepdim=True)
        std = encoder_x_seq.std(0, keepdim=True)
        encoder_x_seq = (encoder_x_seq - mean) / (std + 1e-6)

        encoder_x_len = encoder_x_seq.shape[0]
        encoder_len_diff = encoder_seq_len - encoder_x_len

        tokens = tokenize(str(df['Normalized_Transcription'].iloc[i]))

        decoder_x_len = len(tokens) + 1
        decoder_len_diff = decoder_seq_len - decoder_x_len

        if encoder_len_diff >= 0 and decoder_len_diff >= 0:
            encoder_x.append(
                nn.functional.pad(encoder_x_seq, (0, 0, 0, encoder_len_diff))
            )
            src_padding_mask.append(
                torch.arange(encoder_seq_len) >= encoder_x_len
            )

            decoder_x.append(
                torch.zeros(decoder_seq_len, dtype=torch.long)
            )
            decoder_x[-1][0] = token_list['<BOS>']

            targets.append(
                torch.full((decoder_seq_len,), -1, dtype=torch.long)
            )

            for j, token in enumerate(tokens):
                token_id = token_list[token]
                decoder_x[-1][j+1] = token_id
                targets[-1][j] = token_id

            targets[-1][decoder_x_len-1] = token_list['<EOS>']

            tgt_padding_mask.append(
                torch.arange(decoder_seq_len) >= decoder_x_len
            )
            tgt_casual_mask.append(
                torch.triu(
                    torch.ones(decoder_seq_len, decoder_seq_len, dtype=torch.bool),
                    diagonal=1
                )
            )

            if i % batches == 0 or i == count - 1:
                for i in range(10000):
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

                    optimizer.step()
                    scheduler.step()

                    batch_num = i // batches if i % batches == 0 else i // batches + 1

                    print(f"Batch {batch_num} Loss: {loss.item():.4f}")

                encoder_x.clear()
                decoder_x.clear()
                src_padding_mask.clear()
                tgt_padding_mask.clear()
                tgt_casual_mask.clear()
                targets.clear()

n_mels = 128
vocab_size = len(token_list)
d_model = 512
encoder_seq_len = 500
decoder_seq_len = 50
h = 8
d_k = d_v = d_model // h
d_ff = 2048
n = 6
dropout = 0.1

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
    list((dir/'wavs').iterdir()),
    n_mels,
    encoder_seq_len,
    decoder_seq_len,
    model,
    criterion,
    optimizer,
    scheduler,
    batches
)
