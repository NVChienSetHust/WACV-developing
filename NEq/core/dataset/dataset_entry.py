import torchvision
import pyvww
import torch
import torch.utils.data as data_utils
from typing import Tuple

from .vision import *
from ..utils.config import config
from .vision.transform import *

__all__ = ["build_dataset"]


class MapDataset(data_utils.Dataset):
    """Given a dataset, creates a dataset which applies a mapping function to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.

    Args:
        dataset:
        map_fn:
    """

    def __init__(self, dataset, map_fn, with_target=False):
        self.dataset = dataset
        self.map = map_fn
        self.with_target = with_target

    def __getitem__(self, index):
        if self.with_target:
            return self.map(self.dataset[index][0], self.dataset[index][1])
        else:
            return self.map(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def split_dataset(
    dataset: torch.utils.data.Dataset, 
    val_percentage: float = 0.2,
    velocity_len: int = 10
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Randomly splits a `torch.utils.data.Dataset` instance in two non-overlapping separated `Datasets`.

    The split of the elements of the original `Dataset` is based on `val_percentage` or `val_len`.

    Args:
        dataset (torch.utils.data.Dataset): `torch.utils.data.Dataset` instance to be split.
        val_percentage (float): Percentage of elements to be contained in the validation dataset.
        val_len (int): Number of elements of `dataset` contained in the second dataset if `val_percentage` is not provided.

    Returns:
        tuple: A tuple containing the two new datasets.
    """
    dataset_size = len(dataset)
    
    val_size = int(val_percentage * dataset_size)
    train_size = dataset_size - val_size
    velocity_size = velocity_len

    # Split dataset into training and remaining
    train_dataset, remaining_dataset = data_utils.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.manual_seed)
    )

    # Take the first `velocity_len` samples from the training set for the velocity set
    velocity_dataset = torch.utils.data.Subset(train_dataset, range(velocity_len))
    
    # Remove those samples from the training set
    train_dataset = torch.utils.data.Subset(train_dataset, range(velocity_len, train_size))

    # Remaining samples in the remaining_dataset will be used for validation
    valid_dataset = remaining_dataset


    return train_dataset, velocity_dataset, valid_dataset


def build_dataset():
    if config.data_provider.dataset == "image_folder":
        train_dataset, test = image_folder(
            root=config.data_provider.root,
            transforms=ImageTransform(),
        )

    elif config.data_provider.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            config.data_provider.root,
            train=True,
            transform=None,
            download=True,
            )
        test = torchvision.datasets.CIFAR10(
                config.data_provider.root,
                train=False,
                transform=ImageTransform()["val"],
                download=True,
            )

    elif config.data_provider.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            config.data_provider.root,
            train=True,
            transform=None,
            download=True,
        )
        test = torchvision.datasets.CIFAR100(
            config.data_provider.root,
            train=False,
            transform=ImageTransform()["val"],
            download=True,
        )

    elif config.data_provider.dataset == "vww":
        
        train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root="/home/vanchien/on_device_learning/data/datasets/mscoco-dataset/all2014",
            annFile="/home/vanchien/on_device_learning/data/datasets/vww-dataset/annotations/instances_train.json",

            transform=None,
        )
        test = pyvww.pytorch.VisualWakeWordsClassification(
            root="/home/vanchien/on_device_learning/data/datasets/mscoco-dataset/all2014",
            annFile="/home/vanchien/on_device_learning/data/datasets/vww-dataset/annotations/instances_val.json",
            transform=ImageTransform()["val"],
        )

    else:
        raise NotImplementedError(config.data_provider.dataset)

    # These operations allows for the creation of a small validation dataset from which to compute velocities
    train, velocity, validation = split_dataset(train_dataset)
    train, velocity, validation = MapDataset(train, ImageTransform()["train"]), MapDataset(
        velocity, ImageTransform()["val"]), MapDataset(
        validation, ImageTransform()["val"]
    )

    return {"train": train, "vel": velocity, "val": validation, "test": test}
