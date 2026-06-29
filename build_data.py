from pathlib import Path
import pandas as pd

def build_df():
    root = Path('LibriSpeech/train-clean-100')

    rows = []

    for file in root.rglob('*.trans.txt'):
        folder = file.parent

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                utt_id, transcript = line.strip().split(' ', 1)
                path = folder/f'{utt_id}.flac'

                if path.exists():
                    rows.append({
                        'id': utt_id,
                        'path': str(path),
                        'transcript': transcript
                    })

    df = pd.DataFrame(rows)
    df.to_parquet('data.parquet', index=False)

    return df

build_df()
