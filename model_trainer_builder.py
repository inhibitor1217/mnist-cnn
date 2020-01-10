from typing import Tuple

from dotmap import DotMap
from keras import Model

from base.base_data_loader import BaseDataLoader
from base.base_trainer import BaseTrainer
from models.cnn_model import CNNModel
from models.with_load_weights import WithLoadWeights
from trainers.cnn_trainer import CNNTrainer

def build_model_and_trainer(config: DotMap, data_loader: BaseDataLoader) -> Tuple[Model, BaseTrainer]:
    
    model_builder = CNNModel(config)

    model, parallel_model = WithLoadWeights(model_builder, model_name='cnn') \
        .build_model(model_name='cnn')

    trainer = CNNTrainer(model=model, parallel_model=parallel_model, data_loader=data_loader, config=config)

    return model, trainer