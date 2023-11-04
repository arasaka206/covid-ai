from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import Tuple


class CovidDataModule(LightningDataModule):
    def __init__(self,
                 image_sise: Tuple[int] = (512, 512),
                 train_dir: str = "data/train",
                 val_dir: str = "data/val",
                 test_dir: str = "data/test",
                 batch_size: int = 64,
                 num_workers: int = 8):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.image_sise = image_sise
        self.num_workers = num_workers




    def setup(self, stage: str = None):
        self.train_dataset = ImageFolder(root=self.train_dir,
                                         transform=transforms.Compose([
                                             transforms.PILToTensor(),
                                             transforms.RandAugment(),
                                             transforms.ConvertImageDtype(
                                                 torch.float),
                                             transforms.Resize(
                                                 self.image_sise),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                         ]))
        self.val_dataset = ImageFolder(root=self.val_dir,
                                       transform=transforms.Compose([
                                           transforms.PILToTensor(),
                                           transforms.ConvertImageDtype(
                                               torch.float),
                                           transforms.Resize(self.image_sise),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                       ]))
        self.test_dataset = ImageFolder(root=self.test_dir,
                                        transform=transforms.Compose([
                                            transforms.PILToTensor(),
                                            transforms.ConvertImageDtype(
                                                torch.float),
                                            transforms.Resize(self.image_sise),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                        ]))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

