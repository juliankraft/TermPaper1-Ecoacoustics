import yaml
import os
import pandas as pd
import numpy as np
from typing import Callable

class Eval:
    """
    Class to evaluate a hyper parameter tuning run.

    """
    def __init__(
            self, 
            run_path: str):
        
        """
        run_path: Path to the run folder containing the models.
        """
        self.run_path = run_path
        self.model_paths = self.get_model_paths()
        self.settings = self.get_settings()
        self.train_log = self.get_train_log()
        self.predictions = self.get_predictions()
        self.grouped_train_log = self.combined_train_log()
        self.accuracy = self.accuracy()

    def get_model_paths(self):
        labels = [name for name in os.listdir(self.run_path) if os.path.isdir(os.path.join(self.run_path, name))]
        paths = [run_path + label for label in labels]
        filtered_paths = [path for path in paths if not path.endswith('/evaluation')]
        return filtered_paths
    
    def get_settings(self):

        def model_data(model_path):

            path = f'{model_path}/all_parameters.yaml'

            with open(path, 'r') as file:
                model_info = yaml.safe_load(file)
            
            if model_info['n_mels'] is None:
                model_info['transform'] = 'Spec'
            else:
                model_info['transform'] = 'MelSpec'

            return model_info
        
        data = pd.DataFrame()

        for model_path in self.model_paths:
            new_data = model_data(model_path)
            new_row = pd.DataFrame([new_data])
            data = pd.concat([data, new_row], ignore_index=True)
        
        return data
    
    def get_predictions(self, eval_subset: list = ["test"]):
        """
        eval_subset:List of data sets to be extracted. Options are: ["train", "validation", "test"]
        """
        def model_data(model_path, eval_subset=eval_subset):
            for data_set in eval_subset:
                if data_set not in ['train', 'validation', 'test']:
                    raise ValueError(f'{data_set} is not a valid data set. Please choose from: ["train", "validation", "test"]')

            path = f'{model_path}/predictions.csv'
            data = pd.read_csv(path)

            filtered_data = data[data['data_set'].isin(eval_subset)]

            return filtered_data
        
        data = []
        
        for model_path in self.model_paths:
            data.append(model_data(model_path))
        
        return data

    def get_train_log(self):
            
        def model_data(model_path):
            path = f'{model_path}/metrics.csv'
            metrics = pd.read_csv(path)

            return metrics
        
        data = []

        for model_path in self.model_paths:
            data.append(model_data(model_path))
        
        return data
    
    def combined_train_log(self):

        data = pd.DataFrame()

        for i, logs in enumerate(self.train_log):
            logs['model'] = i
            data = pd.concat([data, logs], ignore_index=True)

        data = data.groupby('model')

        return data


    def get_metrics(self, function: Callable[[list, list], list|int]):
        """
        function: Function to be applied to the metrics. The function should take y_true and y_pred as inputs and return a list of metrics.

        Example:
        self.get_metrics(lambda x, y: [np.mean(x == y)], eval_subset=["test"])
        """
        
        metrics = []

        for data in self.predictions:
            metric = function(data['class_ID'], data['class_ID_pred'])
            metrics.append(metric)

        return metrics

    def accuracy(self):

        accuracy = self.get_metrics(lambda x, y: np.mean(x == y))

        return accuracy
