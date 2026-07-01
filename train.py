import torch

def train(
    epochs,
    loader,
    model,
    criterion,
    optimizer,
    scaler,
    device
):
    tgt_causal_mask = torch.triu(
        torch.ones(
            model.decoder_seq_len, model.decoder_seq_len, dtype=torch.bool, device=device
        ),
        diagonal=1
    )

    batch_num = 0

    for i in range(epochs):
        for batch in loader:
            (
                encoder_x_batch,
                decoder_x_batch,
                src_padding_mask_batch,
                tgt_padding_mask_batch,
                targets_batch
            ) = batch

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
                    logits.view(-1, model.vocab_size),
                    targets_batch.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"Epoch: {i+1} | Batch: {batch_num} | Loss: {loss.item():.4f}")

            batch_num += 1

        batch_num = 0

    torch.save(model.state_dict(), "model.pt")
