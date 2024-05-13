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
            class_ids: list[int],
            min_len_in_seconds: int = 1,
            max_len_in_seconds: int = 5):
        """
        
        min_len_in_seconds: -1 to not subset
        """

        self.data = data
        self.transform = transform
        self.class_mapping = {cl: i for i, cl in enumerate(class_ids)}
        self.num_classes = len(self.class_mapping)

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

        class_id = self.class_mapping[class_id]

        waveform = self.get_random_part_padded(waveform=waveform, samplerate=samplerate)

        spectrogram: Tensor = self.transform(waveform[0, :]) 

        spectrogram = (spectrogram + 40) / 40

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
            win_length: int | None = None,
            hop_length: int = 128,
            top_db: int = 80,
            n_mels: int = 128,
            batch_size: int = 8,
            train_min_len_in_seconds: int = 1,
            train_max_len_in_seconds: int = 10,
            eval_max_len_in_seconds: int = 5,
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
        self.sample_rate = 44100

        mel_transform = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_max=self.sample_rate / 2)

        db_transform = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
        self.transform = torch.nn.Sequential(mel_transform, db_transform)
        # self.transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        train_data = self.get_data(training_mode='train')
        train_meta = train_data.get_metadata()
        class_ids, class_count = np.unique(train_meta.class_id, return_counts=True)
        self.class_weights = class_count.sum() / (class_count * class_count.sum())

    def get_data(self, training_mode: str) -> InsectData:
        if training_mode == 'train':
            min_len_in_seconds = self.train_min_len_in_seconds
            max_len_in_seconds = self.train_max_len_in_seconds
        elif training_mode in ['validation', 'test', 'predict']:
            min_len_in_seconds = -1
            max_len_in_seconds = self.eval_max_len_in_seconds
        else:
            raise ValueError('training_mode must be one of: train, validation, test, predict')

        if training_mode == 'predict':
            csv = self.csv
        else:
            csv = self.csv[self.csv.data_set == training_mode]

        data_set = InsectData(
            data=csv,
            transform=self.transform,
            class_ids=self.class_IDs,
            min_len_in_seconds=min_len_in_seconds,
            max_len_in_seconds=max_len_in_seconds,
        )

        return data_set

    def train_dataloader(self) -> TRAIN_DATALOADERS: # Defines how the Train Dataloader is built

        data_set = self.get_data(training_mode='train')

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS: # Defines how the Validation Dataloader is built

        data_set = self.get_data(training_mode='validation')

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS: # Defines how the Test Dataloader is built

        data_set = self.get_data(training_mode='test')

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS: # Defines a Dataloader with all the Data

        csv = self.csv
        data_set = self.get_data(training_mode='test')

        return DataLoader(data_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
