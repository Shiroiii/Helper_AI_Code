from typing import Tuple, List, Dict
from torch import nn
import torch
from tqdm.auto import tqdm
from utils import save_model

def train_step(
        model:nn.Module,
        optimizer:torch.optim.Optimizer,
        loss_fn:nn.Module,
        dataloader:torch.utils.data.DataLoader,
        device:torch.device,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

     Turns a target PyTorch model to training mode and then
     runs through all the required training steps (forward pass, loss calculation, optimizer step).

     Args:
       model: A PyTorch model to be trained.
       dataloader: A DataLoader instance for the model to be trained on.
       loss_fn: A PyTorch loss function to minimize.
       optimizer: A PyTorch optimizer to help minimize the loss function.
       device: A target device to compute on (e.g. "cuda" or "cpu").

     Returns:
       A tuple of training loss and training accuracy metrics.
       In the form (train_loss, train_accuracy). For example:

       (0.1112, 0.8743)
     """
    # Move model to device
    model.to(device)

    # Put model in training mode
    model.train()

    # Setup train loss and accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (imgs, lbls) in enumerate(dataloader):
        imgs, lbls = imgs.to(device), lbls.to(device)

        # Forward pass
        outputs = model(imgs)

        # Calculate the loss
        loss = loss_fn(outputs, lbls)

        # Accumulate training loss
        train_loss += loss.item()

        # Zero gradients
        optimizer.zero_grad()

        # Backward Pass
        loss.backward()

        # Optimizer Step
        optimizer.step()

        # calculate and accumulate the accuracy
        lbl_pred_class = torch.argmax(outputs, dim=1)
        train_acc += (lbl_pred_class == lbls).sum().item()/len(outputs)

    # Get average metrics per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(
        model:nn.Module,
        loss_fn:nn.Module,
        dataloader:torch.utils.data.DataLoader,
        device:torch.device
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

      Turns a target PyTorch model to "eval" mode and then performs
      a forward pass on a testing dataset.

      Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

      Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    # Move model to device
    model.to(device)

    # Set model to eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager:
    with torch.inference_mode():
        # Loop through dataloader batches
        for batch, (imgs, lbls) in enumerate(dataloader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            # Forward Pass
            outputs = model(imgs)

            # Calculate loss
            loss = loss_fn(outputs, lbls)

            # accumulate the loss
            test_loss += loss.item()

            # calculate and accumulate the accuracy
            lbl_pred_class = torch.argmax(outputs, dim=1)
            test_acc += (lbl_pred_class == lbls).sum().item() / len(outputs)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(
        model:nn.Module,
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        optimizer:torch.optim.Optimizer,
        loss_fn:nn.Module,
        device:torch.device,
        writer:torch.utils.tensorboard.writer.SummaryWriter=None,
        epochs:int=5
) -> Dict[str, list]:
    """Trains and tests a PyTorch model.

      Passes a target PyTorch models through train_step() and test_step()
      functions for a number of epochs, training and testing the model
      in the same epoch loop.

      Calculates, prints and stores evaluation metrics throughout.

      Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

      Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                      train_acc: [...],
                      test_loss: [...],
                      test_acc: [...]}
        For example if training for epochs=2:
                     {train_loss: [2.0616, 1.0537],
                      train_acc: [0.3945, 0.3945],
                      test_loss: [1.2641, 1.5706],
                      test_acc: [0.3400, 0.2973]}
    """

    # Create an empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    best_loss = None
    best_acc = None

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        # TRAIN STEP
        train_loss, train_acc = train_step(model=model, optimizer=optimizer, loss_fn=loss_fn,
                                           dataloader=train_dataloader, device=device)
        # TEST STEP
        test_loss, test_acc = test_step(model=model, loss_fn=loss_fn, dataloader=test_dataloader, device=device)

        print(f"Epoch: {epoch+1}", f"train_loss: {train_loss:.4f}", f"test_loss: {test_loss:.4f}", f"train_acc: {train_acc:.4f}", f"test_acc: {test_acc:.4f}", sep=" | ")

        # See if there's writer, if so log to it
        if writer:
            # Add loss results to SummaryWriter
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={
                "train_loss": train_loss,
                "test_loss": test_loss
            }, global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={
                "train_acc": train_acc,
                "test_acc": test_acc
            }, global_step=epoch)


            writer.close()

        # Model checkpointing
        if best_loss is None and best_acc is None:
            best_loss = test_loss
            best_acc = test_acc
            save_model(model=model, target_dir="./model_params", model_name=model.name)

        if test_loss < best_loss and test_acc < best_acc:
            best_loss = test_loss
            best_acc = test_acc
            save_model(model=model, target_dir="./model_params", model_name=model.name)


        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def eval(self, testDataloader: torch.utils.data.DataLoader) -> dict:
    best_model_state = torch.load(f"./model_params/{self.filepath}")["network_params"]
    self.model.load_state_dict(best_model_state)
    self.result = dict()

    self.model.eval()
    with torch.inference_mode():
        for test_images, test_labels in testDataloader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)

            test_pred = self.model(test_images)
            tloss = self.loss(test_pred, test_labels)
            test_loss += tloss

            # test_accuracy += accuracy_score(torch.argmax(test_pred, dim=1).detach().cpu().numpy(), test_labels.cpu().numpy())
            test_accuracy += self.accuracy(test_pred, test_labels)

        self.result["model"] = str(type(self.model)).split("'")[1].split(".")[1]
        self.result["loss"] = round((test_loss / len(testDataloader)).item(), 3)
        self.result["accuracy"] = round((test_accuracy / len(testDataloader)).item(), 3)