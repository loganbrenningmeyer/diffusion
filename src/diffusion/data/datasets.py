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
    sample_shape = get_sample_shape(data_config.dataset)

    # ----------
    # Define Base Transform
    # ----------
    base_transform = [
        T.Resize((sample_shape[1], sample_shape[2])),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ]

    # ----------
    # Load Dataset
    # ----------
    if data_config.dataset == "cifar10":
        norm = [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = T.Compose(base_transform + norm)

        dataset = datasets.CIFAR10(
            root=data_config.data_dir,
            train=True,
            download=True,
            transform=transform
        )

    # ----------
    # Create Subset
    # ----------
    if data_config.subset.enable:
        subset_size = math.floor(len(dataset) * data_config.subset.ratio)
        dataset = Subset(dataset, range(subset_size))

    return dataset


def get_sample_shape(dataset: str):
    """
    
    
    Args:
    
    
    Returns:
    
    """
    if dataset == "cifar10":
        return (3, 32, 32)