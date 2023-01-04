import torch
from torch import nn
from torchvision.models import vgg11, VGG11_Weights, efficientnet_b0, EfficientNet_B0_Weights


class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

      See the original architecture here: https://poloclub.github.io/cnn-explainer/

      Args:
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers.
        output_shape: An integer indicating number of output units.
    """

    def __init__(self, input_channel: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units*25,
                out_features=output_shape
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.fc_layer(x)
        # print(x)
        return x


def create_vgg11(device: torch.device):
    # Get base model and pretrained weights and send to device
    weights = VGG11_Weights.DEFAULT
    model = vgg11(weights=weights).to(device)

    # Freeze the parameters of the base model
    for params in model.features.parameters():
        params.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=25088, out_features=10)
    ).to(device)
    model.name = "VGG11"

    transforms = weights.transforms()

    print(f"[INFO] Created new {model.name} model")

    return model, transforms


def create_effb0(device: torch.device):
    # Get base model and pretrained weights and send to device
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights).to(device)

    # Freeze the parameters of the base model
    for params in model.features.parameters():
        params.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1280, out_features=10)
    ).to(device)
    model.name = "VGG11"

    transforms = weights.transforms()

    print(f"[INFO] Created new {model.name} model")

    return model, transforms
