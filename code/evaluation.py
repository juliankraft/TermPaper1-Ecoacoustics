import yaml
import os
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.metrics import f1_score

class RunEval:
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
        self.predictions = self.get_predictions_data()
        self.grouped_train_log = self.combined_train_log()
        self.accuracy = self.get_accuracy()
        self.f1 = self.get_f1()
        self.f1_per_class = self.get_f1_per_class()
        self.summary = self.get_summary()

    def get_model_paths(self):
        labels = [name for name in os.listdir(self.run_path) if os.path.isdir(os.path.join(self.run_path, name))]
        paths = [self.run_path + label for label in labels]
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
    
    def get_predictions_data(self, eval_subset: list = ["test"]):
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
    
    def get_predictions(self, index: int, eval_subset: list = ["test"]):
        """
        get y_true and y_pred.
        """
        data = self.predictions[index]
        y_true = data['class_ID']
        y_pred = data['class_ID_pred']

        return y_true, y_pred


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


    def get_metrics(self, function: Callable[[list[int], list[int]], np.ndarray|float]):
        """
        function: Function to be applied to the metrics. The function should take y_true and y_pred as inputs and return a list of metrics.

        Example:
        self.get_metrics(lambda x, y: [np.mean(x == y)], eval_subset=["test"])
        """
        
        metrics = []
        df = pd.DataFrame()
        

        for data in self.predictions:
            new_data = function(data['class_ID'], data['class_ID_pred'])

            if isinstance(new_data, np.ndarray):
                new_row = pd.DataFrame([new_data])
                df = pd.concat([df, new_row], ignore_index=True)

            elif isinstance(new_data, float):
                metrics.append(new_data)

            else:
                raise ValueError(f'{new_data} is of the type {type(new_data)} Please return a list or a float.')    
            
        if len(metrics) == 0:
            df.columns = range(32)
            return df
        else:
            return metrics


    def get_accuracy(self):

        accuracy = self.get_metrics(lambda x, y: np.mean(x == y))

        return accuracy
    
    def get_f1(self):
            
            f1 = self.get_metrics(lambda x, y: f1_score(x, y, average='macro', sample_weight=None, zero_division='warn'))
    
            return f1
    
    def get_f1_per_class(self):
            
            f1_per_class = self.get_metrics(lambda x, y: f1_score(x, y, average=None))
    
            return f1_per_class
    
    def get_best_model(self, metric: str = 'accuracy'):

        if metric == 'accuracy':

            metrics = self.accuracy

        else:
            raise ValueError(f'{metric} is not yet implemented.')
        
        max_value = max(metrics)
        max_index = metrics.index(max_value)

        return max_index
    
    def get_summary(self):

        pd.set_option('future.no_silent_downcasting', True)

        data = self.settings
        data['accuracy'] = self.accuracy
        data['f1'] = self.f1
        data['trained_epochs'] = self.grouped_train_log.max()["epoch"]
        data['num_trainable_params'] = self.settings['num_trainable_params']
        data['n_mels'] = data['n_mels'].fillna(-1).astype(int)
        data = data.sort_values(by='accuracy', ascending=False)

        return data
    