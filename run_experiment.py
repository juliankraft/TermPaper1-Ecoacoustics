"""Run experiments with different hyperparameters.

Usage:
    python run_experiment.py [--overwrite] [--dryrun] [--dev] [--hp_ids HP_IDS]

Example:
    python run_experiment.py --hp_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

"""

import os
from argparse import ArgumentParser
import itertools

# Define hyperparameters
n_mels_options = [64, -1]
n_res_blocks_options = [2, 3, 4]
learning_rate_options = [0.001, 0.0001]
kernel_size_options = [3, 5, 7]




if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing run')
    parser.add_argument('--dryrun', action='store_true', help='Print commands without executing')
    parser.add_argument('--dev', action='store_true', help='Quick dev run.')
    # parser.add_argument('--hp_ids', type=int, nargs='+', default=-1, help='HPs to evaluate')

    args = parser.parse_args()

    # creating params list
    combinations = list(itertools.product(n_mels_options, n_res_blocks_options, learning_rate_options, kernel_size_options))

    PARAM_LIST = [f'--n_mels {n_mels} --n_res_blocks {n_res_blocks} --learning_rate {learning_rate} --kernel_size {kernel_size}' 
        for n_mels, n_res_blocks, learning_rate, kernel_size in combinations]
    
    print("###########################################################################################")
    print("################################# New Run Starting ########################################")
    print("###########################################################################################")
    print("The following hyperparameter combinations can be evaluated:")
    for index, params in enumerate(PARAM_LIST):
        print(f'{index}: {params}')
    print("Select wich part of the list you want to evaluate.")

    start_index_input = input("Enter the start index (default is 0): ")
    stop_index_input = input(f"Enter the stop index (default is {len(PARAM_LIST) - 1}): ")

    # Set default values if the input is empty
    start_index = int(start_index_input) if start_index_input.strip() else 0
    stop_index = int(stop_index_input) if stop_index_input.strip() else len(PARAM_LIST) - 1


    hp_ids = range(start_index, stop_index + 1)

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
            print("###########################################################################################")
            print(command)
            print("###########################################################################################")
            os.system(command)
