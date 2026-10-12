import json
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_data_loaders
from src.model import CNN_FC as DigitClassifier


def train_one_epoch(training_dataloader, epoch_index, writer):
    report_interval = len(training_dataloader) // 10

    running_loss = 0.
    last_loss = 0.
    for i, batch in enumerate(training_dataloader):
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % report_interval == report_interval - 1:
            last_loss = running_loss / report_interval # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            x = epoch_index * len(training_dataloader) + i + 1
            writer.add_scalar('loss/train', last_loss, x)
            running_loss = 0.

    return last_loss

def validate_one_epoch(validation_dataloader):
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(validation_dataloader):
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss

    avg_loss = running_loss / (i + 1)

    return avg_loss


if __name__ == "__main__":
    cfg_path = 'cfg.json'
    print(f"Loading configuration from {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    torch.manual_seed(42)

    train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(
        cfg["data"]["dir"],
        cfg["data"]["train_ratio"],
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"]
    )

    model = DigitClassifier(
        cfg["model"]["architecture"]["feature_size"],
        cfg["model"]["architecture"]["feature_map_sizes"],
        cfg["model"]["architecture"]["conv_kernel_sizes"],
        cfg["model"]["architecture"]["pool_kernel_sizes"],
        cfg["model"]["architecture"]["pool_strides"],
        cfg["model"]["architecture"]["fc_sizes"]
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        momentum=cfg["train"]["momentum"]
    )

    loss_fn = torch.nn.functional.cross_entropy

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/digit_classifier_{timestamp}')

    best_vloss = 1_000_000
    for epoch in range(cfg["train"]["max_epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['train']['max_epochs']}")

        model.train(True)
        avg_loss = train_one_epoch(train_dataloader, epoch, writer)
        avg_vloss = validate_one_epoch(valid_dataloader)

        print(f'LOSS train {avg_loss} valid {avg_vloss}')
        writer.add_scalars(
            'loss', {
                'training' : avg_loss,
                'validation' : avg_vloss
            },
            epoch + 1
        )
        
        if avg_loss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'runs/model_{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)
    