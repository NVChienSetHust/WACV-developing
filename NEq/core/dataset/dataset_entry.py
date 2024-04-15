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
    val_len: int = 10
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
    dataset_length = int(len(dataset))
    train_length = int(dataset_length - val_len)
    train_dataset, valid_dataset = data_utils.random_split(
        dataset,
        [train_length, val_len],
        generator=torch.Generator().manual_seed(config.manual_seed),
    )

    return train_dataset, valid_dataset


def build_dataset():
    print("build dataset")
    if config.data_provider.dataset == "image_folder":
        train_dataset, test = image_folder(
            root=config.data_provider.root,
            transforms=ImageTransform(),
        )
    # Flowers 102 has its own validation set
    elif config.data_provider.dataset == "flowers102":
        train, validation_set, test = FLOWERS102(
            root=config.data_provider.root,
            transforms=ImageTransform(),
        )
        if (
            config.data_provider.use_validation_for_velocity
            and config.data_provider.use_validation
        ):
            validation, validation_for_velocity = split_dataset(
                dataset=validation_set
            )  # Take 10 elements from validation_set
            return {
                "train": train,
                "val": validation,
                "test": test,
                "val_velocity": validation_for_velocity,
            }

        elif (
            config.data_provider.use_validation_for_velocity
            and not config.data_provider.use_validation
        ):
            _, validation_for_velocity = split_dataset(
                dataset=validation_set
            )  # Take 10 elements from validation_set

            train = MapDataset(train, ImageTransform()["train"])
            validation_for_velocity = MapDataset(
                validation_for_velocity, ImageTransform()["val"]
            )
            return {
                "train": train,
                "val_velocity": validation_for_velocity,
                "test": test,
            }

        elif (
            not config.data_provider.use_validation_for_velocity
            and config.data_provider.use_validation
        ):
            train = MapDataset(train, ImageTransform()["train"])
            validation = MapDataset(validation_set, ImageTransform()["val"])
            return {"train": train, "val": validation, "test": test}

        elif (
            not config.data_provider.use_validation_for_velocity
            and not config.data_provider.use_validation
        ):
            return {"train": train, "test": test}

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

    if (
        config.data_provider.use_validation_for_velocity
        and config.data_provider.use_validation
    ):
        train, validation_set = split_dataset(
            dataset=train_dataset,
            val_len=int(
                config.data_provider.validation_percentage * len(train_dataset)
            ),
        )  # Divide the train_dataset into train and validation according to a predifined validation_percentage
        validation, validation_for_velocity = split_dataset(
            dataset=validation_set
        )  # Take 10 elements from validation_set

        train = MapDataset(train, ImageTransform()["train"])
        validation = MapDataset(validation, ImageTransform()["val"])
        validation_for_velocity = MapDataset(
            validation_for_velocity, ImageTransform()["val"]
        )
        return {
            "train": train,
            "val": validation,
            "test": test,
            "val_velocity": validation_for_velocity,
        }

    elif (
        config.data_provider.use_validation_for_velocity
        and not config.data_provider.use_validation
    ):
        train, validation_for_velocity = split_dataset(
            dataset=train_dataset
        )  # Take 10 elements from train_dataset

        train = MapDataset(train, ImageTransform()["train"])
        validation_for_velocity = MapDataset(
            validation_for_velocity, ImageTransform()["val"]
        )
        return {"train": train, "val_velocity": validation_for_velocity, "test": test}

    elif (
        not config.data_provider.use_validation_for_velocity
        and config.data_provider.use_validation
    ):
        train, validation = split_dataset(
            dataset=train_dataset,
            val_len=int(
                config.data_provider.validation_percentage * len(train_dataset)
            ),
        )  # Divide the train_dataset into train and validation according to a predifined validation_percentage

        train = MapDataset(train, ImageTransform()["train"])
        validation = MapDataset(validation, ImageTransform()["val"])
        return {"train": train, "val": validation, "test": test}

    elif (
        not config.data_provider.use_validation_for_velocity
        and not config.data_provider.use_validation
    ):
        train = MapDataset(train_dataset, ImageTransform()["train"])
        return {"train": train, "test": test}
