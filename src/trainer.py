import torch


def train_one_batch(
        model,
        optimizer,
        loss_fn,
        batch
    ):
    inputs, labels = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss

def train_one_epoch(
        model,
        optimizer,
        loss_fn,
        dataloader,
        epoch_index,
        *,
        writer = None
    ):
    report_interval = len(dataloader) // 10

    running_loss = 0.
    last_loss = 0.
    for i, batch in enumerate(dataloader):
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % report_interval == report_interval - 1:
            last_loss = running_loss / report_interval
            print(f'  batch {i + 1} loss: {last_loss}')
            if writer:
                x = epoch_index * len(dataloader) + i + 1
                writer.add_scalar('loss/train', last_loss, x)
            running_loss = 0.

    return last_loss

def validate_one_epoch(
        model,
        loss_fn,
        dataloader
    ):
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss

    avg_loss = running_loss / (i + 1)

    return avg_loss