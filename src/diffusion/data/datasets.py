import math
from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as T
from omegaconf import DictConfig


def load_dataset(data_config: DictConfig):
    """
    Loads, applies transforms, and returns the dataset as 
    specified in the given data_config.
    
    Parameters:
        data_config (DictConfig): Config object containing dataset information.
    
    Returns:
        dataset (Dataset): Loaded Dataset object prepared for training. 
    """
    # ----------
    # Define Base Transform
    # ----------
    base_transform = [
        T.Resize(data_config.image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ]

    # ----------
    # Load Dataset
    # ----------
    if data_config.name == "cifar10":
        norm = [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = T.Compose(base_transform + norm)

        dataset = datasets.CIFAR10(
            root=data_config.path,
            train=True,
            download=True,
            transform=transform
        )
        sample_shape = (3, 32, 32)

    # ----------
    # Create Subset
    # ----------
    if data_config.subset.enable:
        subset_size = math.floor(len(dataset) * data_config.subset.ratio)
        dataset = Subset(dataset, range(subset_size))

    return dataset, sample_shape