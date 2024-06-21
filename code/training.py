import os
import shutil
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
# from lightning.pytorch.utilities import seed_everything
import lightning as L
from dataloader import InsectDatamodule
from model import ResNet
import yaml
from argparse import ArgumentParser, Namespace

from utils import PredictionWriter

# initialize the datamodule and the model

def trainer_setup(log_dir: str, args: Namespace, **kwargs) -> tuple[InsectDatamodule, ResNet, Trainer]:

    datamodule = InsectDatamodule(
        csv_paths=args.csv_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        top_db=args.top_db
    )

    resnet = ResNet(
        in_channels=1,
        base_channels=args.base_channels,
        kernel_size=args.kernel_size,
        n_max_pool=args.n_max_pool,
        n_res_blocks=args.n_res_blocks,
        num_classes=datamodule.num_classes,
        learning_rate=args.learning_rate,
        class_weights=datamodule.class_weights)

    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='',
        version='',
    )

    csv_logger = CSVLogger(
        save_dir=log_dir,
        name='',
        version='',
    )

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=args.patience),
            ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best'),
            PredictionWriter(),
        ],
        **kwargs
    )

    return datamodule, resnet, trainer


def fit(datamodule: InsectDatamodule, resnet: ResNet, trainer: Trainer):
    trainer.fit(
        resnet,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader()
    )


def predict(datamodule: InsectDatamodule, resnet: ResNet, trainer: Trainer):
    trainer.predict(resnet, datamodule.predict_dataloader(), return_predictions=False)


def run_setup(args: Namespace) -> str:
    s = ''
    if args.n_mels is None:
        s += 'spec'
    else:
        s += f'mel{args.n_mels:03d}'

    s += f'_nblock{args.n_res_blocks:d}'
    s += f'_lr{args.learning_rate}'
    s += f'_ks{args.kernel_size}'

    log_dir = os.path.join(args.log_dir, args.tag, s)

    if os.path.exists(log_dir):
        if args.predict:
            pass
        elif args.overwrite:
            shutil.rmtree(log_dir)
        else:
            raise FileExistsError(
                f'directory exists, use `--overwrite` to remove existing run: {log_dir}'
            )
    else:
        if args.predict:
            raise FileExistsError(
                f'existing run required with `--predict`, but path does not exist: {log_dir}'
            )

    os.makedirs(log_dir, exist_ok=True)

    # Write parameters to a YAML file
    with open(os.path.join(log_dir, 'all_parameters.yaml'), 'w') as file:
        yaml.dump(args, file)

    return log_dir


if __name__ == '__main__':

    L.seed_everything(57) # set seed for reproducibility

    parser = ArgumentParser()

    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing run')
    parser.add_argument('--predict', action='store_true', help='Predict, skip tuning')
    parser.add_argument('--dev', action='store_true', help='Quick dev run.')

    # log directory
    parser.add_argument('--log_dir', type=str, default='./logs/',
                    help='Directory to save logs')
    parser.add_argument('--tag', type=str, default='default',
                    help='Experiment tag (subdirectory)')

    # select Dataset
    parser.add_argument('--csv_paths', nargs='+', default=['./data/Orthoptera.csv', './data/Cicadidae.csv' ],
                    help='List of paths to CSV files for datasets')

    # parameters
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')

    parser.add_argument('--n_fft', type=int, default=1024, help='Number of FFT components')
    parser.add_argument('--n_mels', type=int, default=-1, help='Number of Mel bands, -1 means Spectrogram')
    parser.add_argument('--top_db', type=int, default=None, help='Top decibel value for Mel spectrograms')

    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')

    parser.add_argument('--base_channels', type=int, default=8, help='Base number of channels')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for convolutions')
    parser.add_argument('--n_max_pool', type=int, default=3, help='Number of max pooling layers')
    parser.add_argument('--n_res_blocks', type=int, default=4, help='Number of residual blocks')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

    args = parser.parse_args()

    if args.n_mels == -1:
        args.n_mels = None

    log_dir = run_setup(args)

    if args.dev:
        trainer_kwargs = {
            'max_epochs': 1,
            'limit_train_batches': 1,
            'limit_val_batches': 1,
            'limit_predict_batches': 1
        }
    else:
        trainer_kwargs = {
            'max_epochs': -1,
        }

    datamodule, resnet, trainer = trainer_setup(log_dir=log_dir, args=args, **trainer_kwargs)

    if not args.predict:
        fit(datamodule, resnet, trainer)

    best_checkpoint = os.path.join(log_dir, 'checkpoints', 'best.ckpt')
    if not os.path.exists(best_checkpoint):
        raise FileNotFoundError(
            f'cannot load best checkpoint: {best_checkpoint}'
        )

    resnet = ResNet.load_from_checkpoint(checkpoint_path=best_checkpoint)

    predict(datamodule, resnet, trainer)
