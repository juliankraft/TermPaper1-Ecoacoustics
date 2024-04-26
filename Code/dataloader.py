# Import the necessary libraries and modules

import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pytorch_lightning as pl

from torch import Tensor
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class InsectData(Dataset):
    """
    Class to generate a dataset of insect sounds

    """

    def __init__(
            self,
            data: pd.DataFrame,
            transform: torch.nn.Module,
            num_classes: int,
            min_len_in_seconds: int = 1,
            max_len_in_seconds: int = 5):
        """
        
        min_len_in_seconds: -1 to not subset
        """

        self.data = data
        self.transform = transform
        self.num_classes = num_classes

        if min_len_in_seconds == -1:
            df = self.get_metadata()
            max_length = df['file_length'].max()
            self.min_len_in_seconds = max_length
            self.max_len_in_seconds = max_length
        else:
            self.min_len_in_seconds = min_len_in_seconds
            self.max_len_in_seconds = max_len_in_seconds

    def get_random_part_padded(self, waveform: Tensor, samplerate: int) -> Tensor:

        min_len_in_samples = int(self.min_len_in_seconds * samplerate)
        max_len_in_samples = int(self.max_len_in_seconds * samplerate)

        part_length = np.random.randint(min_len_in_samples, max_len_in_samples + 1)
        sample_length = waveform.shape[1]

        part_length = min(part_length, sample_length)

        sample_start_index = np.random.randint(0, sample_length - part_length + 1)
        sample_end_index = sample_start_index + part_length

        waveform_part = waveform[:, sample_start_index:sample_end_index]

        actual_part_length = waveform_part.shape[1]
        pad_length = max_len_in_samples - actual_part_length

        waveform_pad = torch.nn.functional.pad(waveform_part, pad=(pad_length, 0, 0, 0))

        return waveform_pad

    def get_metadata(self) -> pd.DataFrame:

        all_metadata = []

        for idx in range(len(self.data)):
            _, _, _, metadata = self.get_single_sample(idx=idx)

            all_metadata.append(metadata)

        return pd.DataFrame(all_metadata)

    def get_single_sample(self, idx: int):
        sample_meta = self.data.iloc[idx]
        data_path = sample_meta.data_path
        species = sample_meta.species
        class_id = sample_meta.class_ID
        file_name = sample_meta.file_name
        data_set = sample_meta.data_set

        path = os.path.join(data_path, species, file_name).replace('\\', '/')

        if not os.path.exists(path):
            raise FileNotFoundError(
                f'file not found: \'{path}\'.'
            )

        waveform, samplerate = self.load_sample(path)

        metadata = {
            'family': file_name,
            'species': species,
            'class_id': class_id,
            'data_set': data_set,
            'file_length': waveform.shape[-1] / samplerate,
            'path': path
        }

        return waveform, samplerate, class_id, metadata

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        
        waveform, samplerate, class_id, _ = self.get_single_sample(idx=idx)

        waveform = self.get_random_part_padded(waveform=waveform, samplerate=samplerate)

        spectrogram: Tensor = self.transform(waveform[0, :]) 

        species_one_hot: Tensor = torch.nn.functional.one_hot(
            torch.as_tensor(class_id, dtype=torch.long),
            num_classes=self.num_classes)

        return spectrogram, species_one_hot

    @staticmethod
    def load_sample(path: str) -> tuple[Tensor, int]:
        waveform, samplerate = torchaudio.load(path)

        return waveform, samplerate


class InsectDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            csv_paths: list[str] | str,
            n_fft: int = 256,
            hop_length: int = 128,
            batch_size: int = 8,
            train_min_len_in_seconds: int = 1,
            train_max_len_in_seconds: int = 10,
            eval_max_len_in_seconds: int = 5,
            use_mel: bool = False,
            num_workers: int = 0):
        super().__init__()

        self.batch_size = batch_size
        self.train_min_len_in_seconds = train_min_len_in_seconds
        self.train_max_len_in_seconds = train_max_len_in_seconds
        self.eval_max_len_in_seconds = eval_max_len_in_seconds

        self.num_workers = num_workers

        csv_paths = [csv_paths] if isinstance(csv_paths, str) else csv_paths # if there is only one csv path passed, it creates a list

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

        if use_mel:
            self.transform = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length)
        else:
            self.transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)

    def train_dataloader(self) -> TRAIN_DATALOADERS: # Defines how the Train Dataloader is built

        csv = self.csv[self.csv.data_set == 'train']

        data_set = InsectData(
            data=csv,
            transform=self.transform,
            num_classes=self.num_classes,
            min_len_in_seconds=self.train_min_len_in_seconds,
            max_len_in_seconds=self.train_max_len_in_seconds,
        )

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS: # Defines how the Validation Dataloader is built

        csv = self.csv[self.csv.data_set == 'validation']

        data_set = InsectData(
            data=csv,
            transform=self.transform,
            num_classes=self.num_classes,
            min_len_in_seconds=-1,
            max_len_in_seconds=self.eval_max_len_in_seconds
        )

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS: # Defines how the Test Dataloader is built

        csv = self.csv[self.csv.data_set == 'test']

        data_set = InsectData(
            data=csv,
            transform=self.transform,
            num_classes=self.num_classes,
            min_len_in_seconds=-1,
            max_len_in_seconds=self.eval_max_len_in_seconds
        )

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS: # Defines a Dataloader with all the Data

        csv = self.csv

        data_set = InsectData(
            data=csv,
            transform=self.transform,
            num_classes=self.num_classes,
            min_len_in_seconds=-1,
            max_len_in_seconds=self.eval_max_len_in_seconds
        )

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
