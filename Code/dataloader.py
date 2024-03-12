# Import the necessary libraries and modules

import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from IPython.display import Audio
import numpy as np
import pytorch_lightning as pl

from torch import Tensor
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class InsectData(Dataset):

    """
    Class to generate a dataset of insect sounds.
    """

    def __init__(self, data: pd.DataFrame, transform: torch.nn.Module, num_classes: int):
        self.data = data
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample_meta = self.data.iloc[idx]
        data_path = sample_meta.data_path
        species = sample_meta.species
        class_id = sample_meta.class_ID
        file_name = sample_meta.file_name

        path = os.path.join(data_path, species, file_name)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f'file not found: \'{path}\'.'
            )

        waveform, sample_rate = torchaudio.load(path)

        spectrogram: Tensor = self.transform(waveform[0, :])

        species_one_hot: Tensor = torch.nn.functional.one_hot(
            torch.as_tensor(class_id, dtype=torch.long),
            num_classes=self.num_classes)

        return spectrogram, species_one_hot


class InsectDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            csv_paths: list[str] | str,
            n_fft: int = 256,
            hop_length: int = 128,
            batch_size: int = 8):
        super().__init__()

        self.batch_size = batch_size

        csv_paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths

        csv_list = []

        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f'`csv_path` does not exist: \'{csv_path}\'.'
                )

            csv = pd.read_csv(csv_path)
            data_path = csv_path.split('.csv')[0]
            csv['data_path'] = data_path

            csv_list.append(csv)

        csv = pd.concat(csv_list)

        self.class_IDs = sorted(csv.class_ID.unique())
        self.num_classes = len(self.class_IDs)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f'`data_path` does not exist: \'{data_path}\'.'
            )

        self.csv = csv

        self.transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        csv = self.csv[self.csv.data_set == 'train']

        data_set = InsectData(
            data=csv, transform=self.transform, num_classes=self.num_classes)

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:

        csv = self.csv[self.csv.data_set == 'validation']

        data_set = InsectData(
            data=csv, transform=self.transform, num_classes=self.num_classes)

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:

        csv = self.csv[self.csv.data_set == 'test']

        data_set = InsectData(
            data=csv, transform=self.transform, num_classes=self.num_classes)

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=False)
