def tokenize(text):
    tokens = []

    for word in text.split():
        token = ''.join(
            char.lower() if char.isalnum() else ''
            for char in word
        )

        if token and token not in tokens:
            tokens.append(token)

    return tokens

def build_vocab(transcripts):
    vocab = {}

    vocab['<PAD>']= 0
    vocab['<BOS>'] = 1
    vocab['<EOS>'] = 2

    next_id = 3

    for transcript in transcripts:
        for token in tokenize(transcript):
            if token and token not in vocab:
                vocab[token] = next_id
                next_id += 1

    return vocab
