from model.model import Model
from data.data import Data
from data.tokenizer import build_vocab
from train import train

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

batch_size = 30

n_fft = 512
hop_len = 128
n_mels = 64
encoder_seq_len = 1024
decoder_seq_len = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = Data(
    torchaudio.datasets.LIBRISPEECH(
        root='.',
        url='train-clean-100',
        download=True
    ),
    n_fft,
    hop_len,
    n_mels,
    encoder_seq_len,
    decoder_seq_len,
    device
)

loader = DataLoader(
data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data.collate_fn
)

d_model = 512
h = 8
d_k = d_v = d_model // h
d_ff = 2048
n = 6
dropout = 0.1

vocab_size = len(data.vocab)

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

model = model.to(device)
model.train()
nn.utils.clip_grad_norm(model.parameters(), 1.0)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9,0.98), eps=1e-9)

warmup_steps = 4000

def lr_lambda(step):
    step = max(step, 1)

    return (d_model ** -0.5) * min(
        step ** -0.5,
        step * (warmup_steps ** -1.5)
    )

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

scaler = torch.amp.GradScaler("cuda")

epochs = 100

train(
    epochs,
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device
)
