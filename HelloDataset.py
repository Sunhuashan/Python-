# PyTorch入门-Dateset
#
# Dataset stores the samples and their corresponding labels，
# and DataLoader wraps an iterable around the Dataset to
# enable easy access to the samples.

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(  # 加载Dataset
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# 遍历和可视化Dataset
label_map = {
    0: 'T-Shirt',
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
