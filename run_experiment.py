import os
from argparse import ArgumentParser

PARAM_LIST = [
    '--n_mels 64 --n_res_blocks 2 --learning_rate 0.001 --kernel_size 3',
    '--n_mels 64 --n_res_blocks 2 --learning_rate 0.0001 --kernel_size 3',
    '--n_mels 64 --n_res_blocks 3 --learning_rate 0.001 --kernel_size 3',
    '--n_mels 64 --n_res_blocks 3 --learning_rate 0.0001 --kernel_size 3',
    '--n_mels -1 --n_res_blocks 2 --learning_rate 0.001 --kernel_size 3',
    '--n_mels -1 --n_res_blocks 2 --learning_rate 0.0001 --kernel_size 3',
    '--n_mels -1 --n_res_blocks 3 --learning_rate 0.001 --kernel_size 3',
    '--n_mels -1 --n_res_blocks 3 --learning_rate 0.0001 --kernel_size 3',
    '--n_mels 64 --n_res_blocks 2 --learning_rate 0.001 --kernel_size 5',
    '--n_mels 64 --n_res_blocks 2 --learning_rate 0.0001 --kernel_size 5',
    '--n_mels 64 --n_res_blocks 3 --learning_rate 0.001 --kernel_size 5',
    '--n_mels 64 --n_res_blocks 3 --learning_rate 0.0001 --kernel_size 5',
    '--n_mels -1 --n_res_blocks 2 --learning_rate 0.001 --kernel_size 5',
    '--n_mels -1 --n_res_blocks 2 --learning_rate 0.0001 --kernel_size 5',
    '--n_mels -1 --n_res_blocks 3 --learning_rate 0.001 --kernel_size 5',
    '--n_mels -1 --n_res_blocks 3 --learning_rate 0.0001 --kernel_size 5',
]


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing run')
    parser.add_argument('--dryrun', action='store_true', help='Print commands without executing')
    parser.add_argument('--dev', action='store_true', help='Quick dev run.')
    parser.add_argument('--hp_ids', type=int, nargs='+', required=True, help='HPs to evaluate')

    args = parser.parse_args()
    hp_ids = args.hp_ids

    num_hp_ids = len(PARAM_LIST)

    commands = []
    for hp_id in hp_ids:
        if not (0 <= hp_id < num_hp_ids):
            raise ValueError(
                f'invalid `hp_id` detected: `hp_id={hp_id}` is out of range (0, {num_hp_ids - 1}).'
            )

        command = f'python code/training.py {PARAM_LIST[hp_id]}'

        if args.overwrite:
            command += ' --overwrite'

        if args.dev:
            command += ' --dev'

        commands.append(command)

    for command in commands:
        if args.dryrun:
            print(command)
        else:
            os.system(command)
