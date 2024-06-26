# loading libraries
import yaml
import csv
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from argparse import ArgumentParser

# Register a constructor for argparse.Namespace tag that returns a dictionary
yaml.SafeLoader.add_constructor(
    'tag:yaml.org,2002:python/object:argparse.Namespace', 
    yaml.SafeLoader.construct_mapping)


# defining some helper functions
def model_paths(run_path):
    labels = [name for name in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, name))]
    paths = [run_path + label for label in labels]
    filtered_paths = [path for path in paths if not path.endswith('/evaluation')]
    return filtered_paths


def get_ytrue_yhat(model_path: str, data_set_selection: list = ["test"]):

    path = f'{model_path}/predictions.csv'
    data = pd.read_csv(path)

    filtered_data = data[data['data_set'].isin(data_set_selection)]

    y_true = filtered_data['class_ID']
    y_pred = filtered_data['class_ID_pred']

    return y_true, y_pred


def drop_keys_from_dict(dictionary, keys_to_drop):
    return {key: value for key, value in dictionary.items() if key not in keys_to_drop}


def get_hp_settings(model_path: str):

    path = f'{model_path}/all_parameters.yaml'

    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    
    keys_to_drop = ['csv_paths', 'dev', 'log_dir', 'overwrite', 'predict', 'tag', 'top_db']

    filtered_data = drop_keys_from_dict(data, keys_to_drop)

    if filtered_data['n_mels'] is None:
        filtered_data['transform'] = 'Spec'
    else:
        filtered_data['transform'] = 'MelSpec'
    
    return filtered_data


def get_trained_epochs(model_path):
    path = f'{model_path}/metrics.csv'
    data = pd.read_csv(path)
    max_epoch = data['epoch'].max().item()
    return max_epoch


def write_dict_to_csv(data, csv_filepath):
    
    file_exists = os.path.isfile(csv_filepath)

    with open(csv_filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        
        if not file_exists:
            # Write the header only if the file doesn't already exist
            writer.writeheader()
        
        # Write the dictionary values
        writer.writerow(data)


#evaluate a trained model
def evaluate_model(path, show: bool = False, save: bool = False):

    metadata = get_hp_settings(path)
    
    y_true, y_pred = get_ytrue_yhat(path)

    f1 = f1_score(y_true, y_pred, average='macro', sample_weight=None, zero_division='warn')
    f1 = round(f1, 3)
    metadata['f1'] = f1

    accuracy = round(np.mean(y_true == y_pred).item(),3)
    metadata['accuracy'] = accuracy

    trained_epochs = get_trained_epochs(path)
    metadata['trained_epochs'] = trained_epochs
    cm_title = os.path.basename(path)
    run_path = os.path.dirname(path)
    cm_folder = f'{run_path}/evaluation/conf_matrix/'
    
    cm_path = f'{cm_folder}{cm_title}.png'

    metadata['cm_path'] = cm_path

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    ax.set_title(cm_title)
    fig.suptitle(f'accuracy: {accuracy}, trained epochs: {trained_epochs}', y=0.1)

    if save:
        os.makedirs(cm_folder, exist_ok=True)
        plt.savefig(cm_path, bbox_inches='tight')

        write_dict_to_csv(metadata, f'{run_path}/evaluation/eval_summary.csv')

    if show:
        plt.show()
        print('### Metadata ###')
        for key, value in metadata.items():
            print(f"{key}: {value}")
    
    plt.close()


def evaluate_all_models(run_path, show: bool = False, save: bool = False, overwrite: bool = False):
    
    print("Running evaluation...")
    
    if save:
        if os.path.exists(f'{run_path}/evaluation'):
            if overwrite:
                shutil.rmtree(f'{run_path}/evaluation')
                print('existing evaluation folder removed')
            else:
                raise Exception('Evaluation folder already exists. Set overwrite=True to overwrite it.')
    
    paths = model_paths(run_path)

    for path in paths:
        try:
            evaluate_model(path, show=show, save=save)
        except FileNotFoundError:
            print(f'No predictions.csv found in {path}')
    
    print('Evaluation completed')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--run_path', type=str, required=True, help='Path to the to the folder')
    parser.add_argument('--show', action='store_true', help='Show the confusion matrix')
    parser.add_argument('--save', action='store_true', help='Save the confusion matrix')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing evaluation folder')
    args = parser.parse_args()

    
    evaluate_all_models(args.run_path, show=args.show, save=args.save, overwrite=args.overwrite)