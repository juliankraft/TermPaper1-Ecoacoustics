"""Run experiments with different hyperparameters.

Usage:
    python run_experiment.py [--overwrite] [--dryrun] [--dev] [--hp_ids HP_IDS]

Example:
    python run_experiment.py --hp_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

"""

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
    '--n_mels -1 --n_res_blocks 2 --learning_rate 0.0001 --kernel_size 5', #13
    '--n_mels -1 --n_res_blocks 3 --learning_rate 0.001 --kernel_size 5',
    '--n_mels -1 --n_res_blocks 3 --learning_rate 0.0001 --kernel_size 5',
    '--n_mels 64 --n_res_blocks 2 --learning_rate 0.001 --kernel_size 7', #16
    '--n_mels 64 --n_res_blocks 2 --learning_rate 0.0001 --kernel_size 7', #17
    '--n_mels 64 --n_res_blocks 3 --learning_rate 0.001 --kernel_size 7', #18
    '--n_mels 64 --n_res_blocks 3 --learning_rate 0.0001 --kernel_size 7', #19
    '--n_mels -1 --n_res_blocks 2 --learning_rate 0.001 --kernel_size 7', #20
    '--n_mels -1 --n_res_blocks 2 --learning_rate 0.0001 --kernel_size 7', #21
    '--n_mels -1 --n_res_blocks 3 --learning_rate 0.001 --kernel_size 7', #22 #repeat
    '--n_mels -1 --n_res_blocks 3 --learning_rate 0.0001 --kernel_size 7' #23 #repeat
]


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing run')
    parser.add_argument('--dryrun', action='store_true', help='Print commands without executing')
    parser.add_argument('--dev', action='store_true', help='Quick dev run.')
    parser.add_argument('--hp_ids', type=int, nargs='+', default=-1, help='HPs to evaluate')

    args = parser.parse_args()
    hp_ids = args.hp_ids

    num_hp_ids = len(PARAM_LIST)

    if args.hp_ids == -1:
        hp_ids = range(num_hp_ids)

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
            print(command)
            os.system(command)
