import os
from datetime import datetime
from typing import Optional, List, Generator, Tuple

import numpy as np
from dotmap import DotMap
import tensorflow as tf
from tensorpack.dataflow import BatchData, DataFlow

from base.base_data_loader import BaseDataLoader
from utils.dataflow import GeneratorToDataFlow
from utils.image import normalize_image

class ProcessingDataFlow(DataFlow):
    def __init__(self, ds: DataFlow, crop_size: Optional[Tuple[int, int]], fit_resize: bool, random_flip: bool,
                 random_brightness: bool, random_contrast: bool,
                 random_fit: bool, random_fit_max_ratio: float = 1.3) -> None:
        self.ds = ds
    
    def reset_state(self) -> None:
        super().reset_state()

        # set random seed per process
        seed = (id(self) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
        np.random.seed(seed)

        self.ds.reset_state()
        
    def __iter__(self):
        for x, y in self.ds.__iter__():
            # normalize
            yield normalize_image(x), y

    def __len__(self):
        return self.ds.__len__()

def load_data() -> DotMap:
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    num_train = train_x.shape[0]
    num_test  = test_x.shape[0]

    train_x = train_x.reshape((num_train, 28, 28, 1))
    train_y_onehot = np.zeros((num_train, 10), dtype=float)
    train_y_onehot[np.arange(num_train), train_y] = 1.

    test_x = test_x.reshape((num_test, 28, 28, 1))
    test_y_onehot = np.zeros((num_test, 10), dtype=float)
    test_y_onehot[np.arange(num_test), test_y] = 1.

    data: DotMap = DotMap({ 
        'train_x': train_x, 
        'train_y': train_y_onehot, 
        'test_x':  test_x, 
        'test_y':  test_y_onehot 
    })

    return data


def data_to_generator(x, y, shuffle: bool) -> Generator:
    while True:
        assert x.shape[0] == y.shape[0]
        data_size = x.shape[0]
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        
        for index in indices:
            yield x[index], y[index]

def data_to_dataflow(x, y, config: DotMap) -> DataFlow:
    dataflow = data_to_generator(x, y, True)
    dataflow = GeneratorToDataFlow(dataflow)
    dataflow = BatchData(dataflow, config.trainer.batch_size)
    dataflow.reset_state()

    return dataflow

class MNISTDataLoader(BaseDataLoader):
    def __init__(self, config: DotMap) -> None:
        super().__init__(config)

        data = load_data()
        
        # create train dataflow
        self.train_dataflow: DataFlow = data_to_dataflow(data.train_x, data.train_y, config)
        self.train_dataflow_size = data.train_x.shape[0]

        # create test dataflow
        self.test_dataflow:  DataFlow = data_to_dataflow(data.test_x,  data.test_y,  config)
        self.test_dataflow_size = data.test_x.shape[0]

    def get_train_data_generator(self) -> Generator:
        return self.train_dataflow.get_data()

    def get_test_data_generator(self) -> Generator:
        return self.test_dataflow.get_data()

    def get_train_data_size(self) -> int:
        return self.train_dataflow_size

    def get_test_data_size(self) -> int:
        return self.test_dataflow_size
