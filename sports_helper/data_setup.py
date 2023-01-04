import os

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def data_setup(
        dir: str,
        transforms: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
) -> tuple:
    """
    Creates training and testing DataLoaders.
    Takes in a training directory and testing directory path and turns them into PyTorch Datasets
    and then into PyTorch DataLoaders.

    Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_data, test_data, train_dataloader, val_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
    train_data, test_data, train_dataloader, val_dataloader, test_dataloader, class_names = \
    create_dataloaders(train_dir=path/to/train_dir, test_dir=path/to/test_dir, transform=some_transform, batch_size=32, num_workers=4)
    """

    # print(NUM_WORKERS, " workers", sep="")

    # train_data = datasets.CIFAR10(
    #     root=dir, train=True, download=False, transform=transforms)
    # test_data = datasets.CIFAR10(
    #     root=dir, train=False, download=False, transform=transforms)
    # val_dataset, test_dataset = split_dataset(test_data)
    dataset = ImageFolder(root=dir, transform=transforms)
    class_map = dataset.class_to_idx

    # Create Dataloaders
    dataloader = DataLoader(
        dataset=dataset, shuffle=True if "train" in dir else False, batch_size=batch_size, num_workers=num_workers)

    return dataset, dataloader, class_map


def split_dataset(dataset: datasets, split_size: float = 0.2, seed: int = 42):
    """Randomly splits a given dataset into two proportions based on split_size and seed.

        Args:
            dataset (torchvision.datasets): A PyTorch Dataset, typically one from torchvision.datasets.
            split_size (float, optional): How much of the dataset should be split?
                E.g. split_size=0.2 means there will be a 20% split and an 80% split. Defaults to 0.2.
            seed (int, optional): Seed for random generator. Defaults to 42.

        Returns:
            tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and
                random_split_2 is of size (1-split_size)*len(dataset).
    """

    # Create split lengths based on original dataset length
    length_1 = int(len(dataset) * split_size)
    length_2 = len(dataset) - length_1

    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: \
            {length_1} ({int(split_size*100)}%), {length_2} ({int((1 - split_size)*100)}%)")

    random_split_1, random_split_2 = torch.utils.data.random_split(
        dataset, lengths=[length_1, length_2], generator=torch.manual_seed(seed))

    return random_split_1, random_split_2
