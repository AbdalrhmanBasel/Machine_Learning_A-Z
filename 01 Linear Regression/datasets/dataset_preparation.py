import numpy as np
import pandas as pd
from easydict import EasyDict


class DatasetPreparation():
    def __init__(self,data_path, cfg: EasyDict):
        advertising_dataframe = pd.read_csv(data_path)
        inputs = np.asarray(advertising_dataframe['inputs'])
        targets = np.asarray(advertising_dataframe['targets'])
        self.__divide_into_sets(inputs, targets, cfg.train_size, cfg.valid_size)

    def __divide_into_sets(self, inputs: np.ndarray, targets: np.ndarray, train_set_percent: float = 0.8,
                           valid_set_percent: float = 0.1) -> None:
        """
        # TODO (DONE): define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid, self.inputs_test, self.targets_test

        This function takes the inputs, targets, training set percentage, and valid set percentages to
        calculate the size of each dataset and than split the dataset into train, valid, and test sets
        accordingly.


        """
        # Calculate sizes of each dataset
        total_size = inputs.shape[0]  # 1000
        train_size = int(total_size * train_set_percent)  # 800
        valid_size = int(total_size * valid_set_percent)  # 100
        test_size = total_size - train_size - valid_size  # 100

        # Split inputs and targets into train, valid, and test sets
        self.inputs_train, self.inputs_valid, self.inputs_test = np.split(inputs, [train_size, train_size + valid_size])
        self.targets_train, self.targets_valid, self.targets_test = np.split(targets, [train_size, train_size + valid_size])

    def __call__(self) -> dict:
        """
        This function returns a dictionary containing the data in the dataset.

        Example to access data:
            # Access the training inputs
            inputs_train = data_dict['inputs']['train']
            print(len(inputs_train))

            # Access the validation targets
            test_valid = data_dict['targets']['test']
            print(len(test_valid))
        """
        return {'inputs': {'train': self.inputs_train,
                           'valid': self.inputs_valid,
                           'test': self.inputs_test},
                'targets': {'train': self.targets_train,
                            'valid': self.targets_valid,
                            'test': self.targets_test}
                }