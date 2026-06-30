def tokenize(text):
    tokens = []

    for word in text.split():
        token = "".join(
            char.lower() if char.isalnum() else ''
            for char in word
        )

        if token and token not in tokens:
            tokens.append(token)

    return tokens

def build_vocab(data):
    vocab = {}

    vocab["<BOS>"] = 0
    vocab["<EOS>"] = 1

    next_id = 2

    for i in range(len(data)):
        _, _, transcript, *_ = data[i]

        for token in tokenize(str(transcript)):
            if token and token not in vocab:
                vocab[token] = next_id
                next_id += 1

    return vocab
