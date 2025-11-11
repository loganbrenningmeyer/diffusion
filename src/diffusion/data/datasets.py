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
    # Define Dataset Transform
    # ----------
    transforms = [
        T.Resize(data_config.image_size),
        T.ToTensor()
    ]

    if data_config.in_ch == 3:
        transforms.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    else:
        transforms.append(T.Normalize((0.5,), (0.5,)))

    transform = T.Compose(transforms)

    # ----------
    # Download Dataset
    # ----------
    if data_config.name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_config.path,
            train=True,
            download=True,
            transform=transform
        )

    return dataset