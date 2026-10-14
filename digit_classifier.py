import json
import logging
from datetime import datetime

import mlflow
import torch
from mlflow.models import infer_signature
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_data_loaders
from src.model import CNN_FC as DigitClassifier
from src.trainer import train_one_epoch, validate_one_epoch

logger = logging.getLogger(__name__)


def setup(cfg_path):
    mlflow.set_experiment("Digit classifier")
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)

    torch.manual_seed(42)

    logger.info(f"Loading configuration from {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    return cfg

def get_optimizer(model, hparams):
    if hparams["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hparams["learning_rate"],
            momentum=hparams["momentum"]
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {hparams['optimizer']}")

    return optimizer

def train(
    model,
    optimizer,
    loss_fn,
    dataloaders,
    hparams,
    *,
    input_example=None,
    signature = None,
    writer = None
):
    with mlflow.start_run() as run:
        mlflow.log_params(hparams)
        
        # best_vloss = 1_000_000
        for epoch in range(hparams["max_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{hparams['max_epochs']}")

            model.train(True)
            avg_loss = train_one_epoch(
                model,
                optimizer,
                loss_fn,
                dataloaders["train"],
                epoch,
                writer=writer
            )
            avg_vloss = validate_one_epoch(
                model,
                loss_fn,
                dataloaders["validation"],
            )

            logger.info(f'LOSS train {avg_loss} valid {avg_vloss}')
            if writer:
                writer.add_scalars(
                    'loss', {
                        'training' : avg_loss,
                        'validation' : avg_vloss
                    },
                    epoch + 1
                )
            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "val_loss": avg_vloss,
                },
                step=epoch,
            )

            if input_example is not None and signature:
                mlflow.pytorch.log_model(
                    model,
                    name=f"checkpoint_{epoch}",
                    input_example=input_example,
                    signature=signature
                )

        if input_example is not None and signature:
            mlflow.pytorch.log_model(
                model,
                name="final_model",
                input_example=input_example,
                signature=signature
            )

def test(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    logger.info(f'Test loss {avg_loss}')
    return avg_loss

def main():
    logging.basicConfig(level=logging.INFO)
    cfg = setup('cfg.json')

    data_cfg = cfg["data"]
    hyperparameters = cfg["hyperparameters"]
    dataloaders = get_data_loaders(
        data_cfg["dir"],
        hyperparameters["splits"],
        hyperparameters["batch_size"],
        data_cfg["num_workers"]
    )

    arch_cfg = cfg["model"]["architecture"]
    model = DigitClassifier(
        arch_cfg["feature_size"],
        arch_cfg["feature_map_sizes"],
        arch_cfg["conv_kernel_sizes"],
        arch_cfg["pool_kernel_sizes"],
        arch_cfg["pool_strides"],
        arch_cfg["fc_sizes"]
    )

    optimizer = get_optimizer(model, hyperparameters)
    loss_fn = torch.nn.functional.cross_entropy

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{timestamp}')
    input_example = torch.randn(32, 1, 28, 28)
    writer.add_graph(model, input_example)
    signature = infer_signature(input_example.numpy(), model(input_example).detach().numpy())

    train(
        model,
        optimizer,
        loss_fn,
        dataloaders,
        hyperparameters,
        input_example=input_example,
        signature=signature,
        writer=writer
    )

    loss = test(model, dataloaders["test"], loss_fn)

if __name__ == "__main__":
    main()
    