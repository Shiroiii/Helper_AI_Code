import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from typing import Dict
import matplotlib.pyplot as plt


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pth.tar") or model_name.endswith(".pt") or model_name.endswith(".pt.tar"), \
        "model name does not end with correct extension. One of .pth, .pt, .pth.tar, .pt.tar is allowed"

    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def create_writer(experiment_name: str, model_name: str, extra: str = None) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

        log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

        Where timestamp is the current date in YYYY-MM-DD format.

        Args:
            experiment_name (str): Name of experiment.
            model_name (str): Name of model.
            extra (str, optional): Anything extra to add to the directory. Defaults to None.

        Returns:
            torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

        Example usage:
            # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
            writer = create_writer(experiment_name="data_10_percent",
                                   model_name="effnetb2",
                                   extra="5_epochs")
            # The above is the same as:
            writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    # returns the current data in YYYY-MM-DD format
    timestamp = datetime.now().strftime("%Y-%m-%d")
    folder_name = Path("runs/")

    if extra:
        # Create log directory path
        log_dir = folder_name / timestamp / experiment_name / model_name / extra
    else:
        log_dir = folder_name / timestamp / experiment_name / model_name

    print(f"[INFO] Created a SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=str(log_dir))


def plot(self, history: Dict[str, list]) -> None:
    train_loss = history["train_loss"]
    test_loss = history["test_loss"]
    train_acc = history["train_acc"]
    test_acc = history["test_acc"]

    # fig = make_subplots(rows=1, cols=2)

    # fig.add_trace(go.Scatter(
    #     name="Training Loss",
    #     x=list(range(self.epochs)),
    #     y=train_loss
    #     ), row=1, col=1)
    # fig.add_trace(go.Scatter(
    #     name="Testing Loss",
    #     x=list(range(self.epochs)),
    #     y=test_loss
    # ), row=1, col=1)

    # fig.add_trace(go.Scatter(
    #     name="Training Accuracy",
    #     x=list(range(self.epochs)),
    #     y=train_acc
    # ), row=1, col=2)

    # fig.add_trace(go.Scatter(
    #     name="Testing Accuracy",
    #     x=list(range(self.epochs)),
    #     y=test_acc
    # ), row=1, col=2)

    # fig.update_layout(yaxis_range=[0, np.ceil(train_loss.max())])
    # fig.show()

    # return fig
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].plot(train_loss, label="train_loss")
    ax[0].plot(test_loss, label="test_loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(train_acc, label="train_accuracy")
    ax[1].plot(test_acc, label="test_accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()
