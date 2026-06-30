from .tokenizer import tokenize, build_vocab

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torchaudio.functional as functional


class Data(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        n_fft: int,
        hop_len: int,
        n_mels: int,
        max_sample_len: int,
        max_transcript_len: int,
        device: torch.device
    ):
        self.dataset = dataset
        self.max_sample_len = max_sample_len
        self.max_transcript_len = max_transcript_len
        self.valid_indices = []

        max_waveform_len = (max_sample_len - 1) * hop_len + n_fft

        for i in range(len(dataset)):
            waveform, sr, transcript, *_ = dataset[i]

            waveform_len = waveform.shape[-1]

            tokens = tokenize(transcript)
            transcript_len = len(tokens) + 1

            if (
                waveform_len <= max_waveform_len and
                transcript_len <= max_transcript_len
            ):
                self.valid_indices.append(i)

        transcripts = []

        for index in self.valid_indices:
            _, _, transcript, *_ = dataset[index]

            transcripts.append(transcript)

        self.vocab = build_vocab(transcripts)
        self.device = device

        self.to_mel = transforms.MelSpectrogram(
            sample_rate=8000,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=n_mels
        )

        self.to_db = transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        return self.dataset[self.valid_indices[index]]

    @staticmethod
    def encode(transcript, vocab):
        tokens = tokenize(transcript)

        return torch.tensor([vocab[token] for token in tokens if token])


    def collate_fn(self, batch):
        encoder_x = []
        src_padding_mask = []

        decoder_x = []
        tgt_padding_mask = []

        targets = []

        for waveform, sr, transcript, *_, in batch:
            waveform = functional.resample(waveform, sr, 8000)

            mel = self.to_mel(waveform).to(self.device)
            mel = self.to_db(mel)
            mel = mel.squeeze(0).transpose(0, 1)
            mel = (mel - mel.mean()) / (mel.std() + 1e-9)

            mel_len = mel.shape[0]
            encoder_len_diff = self.max_sample_len - mel_len

            transcript_encoded = self.encode(transcript, self.vocab)

            transcript_len = len(transcript_encoded)
            decoder_len_diff = self.max_transcript_len - (transcript_len + 1)

            encoder_x.append(
                nn.functional.pad(mel, (0, 0, 0, encoder_len_diff))
            )

            src_padding_mask.append(
                (torch.arange(self.max_sample_len) >= mel_len).to(self.device)
            )

            decoder_x.append(
                torch.zeros((self.max_transcript_len), dtype=torch.long, device=self.device)
            )
            decoder_x[-1][0] = self.vocab['<BOS>']

            tgt_padding_mask.append(
                (torch.arange(self.max_transcript_len) >= transcript_len).to(self.device)
            )

            targets.append(
                torch.full((self.max_transcript_len,), -1, dtype=torch.long, device=self.device)
            )

            for i in range(len(transcript_encoded)):
                decoder_x[-1][i+1] = transcript_encoded[i]
                targets[-1][i] = transcript_encoded[i]

                targets[-1][self.max_transcript_len-1] = self.vocab['<EOS>']

        encoder_x_batch = torch.stack(encoder_x)
        decoder_x_batch = torch.stack(decoder_x)

        src_padding_mask_batch = torch.stack(src_padding_mask)
        tgt_padding_mask_batch = torch.stack(tgt_padding_mask)

        targets_batch = torch.stack(targets)

        return (
            encoder_x_batch,
            decoder_x_batch,
            src_padding_mask_batch,
            tgt_padding_mask_batch,
            targets_batch
        )
