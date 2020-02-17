import numpy as np
from torch.utils import data


def train_test_split(dataset: data.Dataset, size: float = 0.2):
    """
    Produces sampler that can be used to split the dataset into training and validation/test set
    using dataload with respective sampler.

    :param dataset:
    :param size:
    :return: Train Sampler, Test Sampler

    example:

    dataset = torchvision.datasets.MNIST('.mnist', train=True, transform=transforms.ToTensor())
    train, validation = train_test_split(dataset, size=0.2)

    train_loader = DataLoader(dataset, sampler=train, batch_size=20)
    validation_loader = DataLoader(dataset, sampler=validation, batch_size=20)

    """
    num = len(dataset)
    indices = list(range(num))
    np.random.shuffle(indices)

    split = int(np.floor(size * num))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data.sampler.SubsetRandomSampler(train_idx)
    test_sampler = data.sampler.SubsetRandomSampler(valid_idx)

    return train_sampler, test_sampler



