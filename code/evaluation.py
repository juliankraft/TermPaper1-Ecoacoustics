import yaml
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import scipy.stats as stats
import torchaudio

from typing import Callable
from dataloader import InsectDatamodule




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
        self.dataloader = self.get_dataloader()
        self.model_paths = self.get_model_paths()
        self.settings = self.get_settings()
        self.train_log = self.get_train_log()
        self.predictions = self.get_predictions_data()
        self.grouped_train_log = self.combined_train_log()
        self.accuracy = self.get_metrics(sel_metric='accuracy')
        self.f1 = self.get_metrics(sel_metric='f1')
        self.f1_per_class = self.get_metrics(sel_metric='f1_per_class')
        self.summary = self.get_summary()
    
    def get_dataloader(self):
        datamodule = InsectDatamodule(csv_paths=['../data/Cicadidae.csv', '../data/Orthoptera.csv'], batch_size=10)
        dataloader = datamodule.predict_dataloader()
        return dataloader

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


    def get_metrics(
            self, 
            sel_metric: str | None = None, 
            operation: Callable[[list[int], list[int]], np.ndarray|float] | None = None, 
            eval_subset: list = ["test"]):
        """
        operation: Function to be applied to the metrics. The operation should take y_true and y_pred as inputs and return a list of metrics.
        sel_metric: chose a implemented metric. Implemented Options are: ["accuracy", "f1", "f1_per_class"]
        eval_subset: List of data sets to be extracted. Options are: ["train", "validation", "test"]

        Example:
        self.get_metrics(lambda x, y: [np.mean(x == y)], eval_subset=["test"])
        """
        
        operations = {
            'accuracy': lambda x, y: np.mean(x == y),
            'f1': lambda x, y: f1_score(x, y, average='macro', sample_weight=None, zero_division='warn'),
            'f1_per_class': lambda x, y: f1_score(x, y, average=None, sample_weight=None, zero_division='warn')
        }

        # Check if either metric or operation is given.
        if not(sel_metric is None or operation is None):
            raise ValueError('Provide only one of the following: metric or operation.')
        elif operation is not None:
            pass
        elif sel_metric is not None:
            operation = operations[sel_metric]
        else:
            raise ValueError('Provide either metric or operation.')
        
        metrics = []
        df = pd.DataFrame()
        
        predictions = self.get_predictions_data(eval_subset=eval_subset)

        for data in predictions:
            new_data = operation(data['class_ID'], data['class_ID_pred'])

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

    
    def get_best_model(self, sel_metric: str = 'accuracy', eval_subset: list = ["test"]):

        metrics = self.get_metrics(sel_metric=sel_metric, eval_subset=eval_subset)
        
        max_value = max(metrics)
        max_index = metrics.index(max_value)

        return max_index
    
    def get_summary(self, eval_subset: list = ["test"]):

        pd.set_option('future.no_silent_downcasting', True)

        data = self.settings
        data['accuracy'] = self.get_metrics(sel_metric='accuracy', eval_subset=eval_subset)
        data['f1'] = self.get_metrics(sel_metric='f1', eval_subset=eval_subset)
        data['trained_epochs'] = self.grouped_train_log.max()["epoch"]
        data['num_trainable_params'] = self.settings['num_trainable_params']
        data['n_mels'] = data['n_mels'].fillna(-1).astype(int)
        data = data.sort_values(by='accuracy', ascending=False)

        return data
    
class LatexObject:

    instances = []

    def __init__(
            self,
            object_type: str,
            label: str = "",
            project_path: str = "../LaTeX/",
            caption: str = "",
            create_object: Callable = None,
            table_size: str = None):
        
        self.object_type = object_type
        self.label = label
        self.project_path = project_path
        self.caption = caption
        self.create_object = create_object
        self.table_size = table_size

        if object_type == "table":
            pass
    
        elif object_type == "figure":

            plt.rcParams['axes.labelsize'] = 18  # Adjust the size of the axis labels
            plt.rcParams['xtick.labelsize'] = 12  # Adjust the size of the x-axis tick labels
            plt.rcParams['ytick.labelsize'] = 12  # Adjust the size of the y-axis tick labels
            plt.rcParams['axes.titlesize'] = 20  # Adjust the size of the title

        else:
            raise NotImplementedError(f"Object type {object_type} not implemented")
        
        LatexObject.instances.append(self)

        self.path = self.get_path(path_type = "project")
        self.latex_lines = self.get_latex_lines()
        self.latex_command = self.get_latex_command()

    
    def get_path(self, path_type):

        latex = f'{self.object_type}s/{self.label}'
        project = f'{self.project_path}/{latex}'
        
        if path_type == "latex":
            return latex
        elif path_type == "project":
            return project
        else:
            raise ValueError(f"{path_type} is not a valid path type. Please choose from: ['latex', 'project']")
        
    def get_latex_lines(
            self, 
            position = "h", 
            object_width = 1.0,
            caption_width = 0.9):
        
        if self.object_type == "figure":
            latex_lines = [
                f'\\begin{{figure}}[{position}]',
                f'\\centering',
                f'\\captionsetup{{width={caption_width}\linewidth}}',
                f'\\includegraphics[width={object_width}\\textwidth]{{{self.get_path(path_type="latex")}.pdf}}',
                f'\\caption{{{self.caption}}}',
                f'\label{{tab:{self.label}}}',
                f'\end{{figure}}'
                ]
        
        elif self.object_type == "table":

            if self.table_size in ['tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize', 'large', 'Large', 'LARGE', 'huge', 'Huge']:
                head = [f'\\{self.table_size}']
                tail = ['\\normalsize']

            else:
                head = []
                tail = []

            latex_lines_head = [
                f'\\begin{{table}}[{position}]',
                f'\\centering',
                f'\\captionsetup{{width={caption_width}\linewidth}}',
                f'\\caption{{{self.caption}}}',
                f'\label{{fig:{self.label}}}',             
            ] + head

            latex_lines_tail = tail + [
                f'\end{{table}}'
            ]

            latex_lines = latex_lines_head + [self.create_object()] + latex_lines_tail
            
        else:
            raise NotImplementedError(f"Object type {self.object_type} not implemented")
        
        return latex_lines
    
    def print_latex_lines(self):
        for line in self.latex_lines:
            print(line)

    def show(self):
        print(f"{self.object_type}: {self.label}")
        print(f"Caption: {self.caption}")
        if self.object_type == "table":
            print(self.create_object())
        elif self.object_type == "figure":
            self.create_object()
            plt.show()
    
    def export(self):
        if self.object_type == "table":
            with open(f"{self.path}.tex", "w") as file:
                for line in self.latex_lines:
                    file.write(f"{line}\n")

        elif self.object_type == "figure":
            self.create_object()
            plt.tight_layout()
            plt.savefig(f"{self.path}.pdf")
            plt.close()

            with open(f"{self.path}.tex", "w") as file:
                for line in self.latex_lines:
                    file.write(f"{line}\n")

    def get_latex_command(self):
        latex_command = f"\\input{{{self.get_path(path_type='latex')}.tex}}\n"
        length = len(latex_command)
        length_middle = len(self.label) + len(self.object_type) + 4
        length_d = length - length_middle
        before = f'%{(length_d//2 - 1)*"="} {self.object_type}: {self.label} {(length - length_middle - length_d//2 - 2)*"="}%\n'
        after = f'%{(length-3)*"="}%'
        return before + latex_command + after

    @classmethod
    def show_all(cls, select_type: list = ["figure"]):
        if isinstance(select_type, str):
            select_type = [select_type]

        for instance in cls.instances:
            if instance.object_type in select_type:
                instance.show()
    
    @classmethod
    def export_all(cls):
        for instance in cls.instances:
            instance.export()

    @classmethod
    def show_all_latex_commands(cls):
        for instance in cls.instances:
            print(f'{instance.label}:')
            print(f'{instance.latex_command}\n')
