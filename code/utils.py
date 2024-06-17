import pandas as np
import os
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):

    def __init__(self):
        super().__init__(write_interval='epoch')

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        pdl = trainer.predict_dataloaders
        csv: pd.DataFrame = pdl.dataset.data
        log_dir = trainer.log_dir
        prediction_file = os.path.join(log_dir, 'predictions.csv')

        csv['class_ID_pred'] = -1

        for pred in predictions:
            y_hats  = pred['y_hat']
            idxs  = pred['idx']

            for y_hat, idx in zip(y_hats, idxs):
                predicted_class_index = np.argmax(y_hat)
                index = idx.item()

                predicted_class = [k for k, v in pdl.dataset.class_mapping.items() if v == predicted_class_index]

                if len(predicted_class) != 1:
                    raise ValueError(
                        f'could not find predicted class index {predicted_class_index} in class mapping.'
                    )

                predicted_class = predicted_class[0]

                csv.iloc[index, csv.columns.get_loc('class_ID_pred')] = predicted_class

        csv.to_csv(prediction_file)
