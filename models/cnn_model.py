from keras import Model, Input
from keras.engine import Layer
from keras.layers import Conv2D, Dropout, MaxPool2D, ReLU, BatchNormalization, Flatten, Dense, Softmax
from keras.optimizers import Adam

from base.base_model import BaseModel

class CNNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def define_model(self, model_name):
        def conv2d(_input, filters, kernel_size, strides, dropout, name_prefix):
            _x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                name=f'{name_prefix}conv')(_input)
            _x = BatchNormalization()(_x)
            if dropout:
                _x = Dropout(rate=0.5, name=f'{name_prefix}dropout')(_x)
            _x = ReLU()(_x)
            _x = MaxPool2D(pool_size=(2, 2))(_x)
            return _x

        dropout = self.config.model.dropout

        _input = Input(shape=(28, 28, 1), name=f'{model_name}_input')

        x = conv2d(_input, filters=4, kernel_size=(4, 4), strides=(1, 1), dropout=dropout, name_prefix='conv_block1_')
        x = conv2d(x, filters=8, kernel_size=(4, 4), strides=(1, 1), dropout=dropout, name_prefix='conv_block2_')

        x = Flatten()(x)
        x = Dense(64)(x)
        x = Dense(10)(x)
        x = Softmax()(x)

        model = Model(inputs=_input, outputs=x, name=model_name)

        return model

    def build_model(self, model_name):
        model = self.define_model(model_name)

        optimizer = Adam(lr=self.config.model.lr, beta_1=self.config.model.beta1,
            beta_2=self.config.model.beta2, clipvalue=self.config.model.clipvalue,
            clipnorm=self.config.model.clipnorm)
        optimizer = self.process_optimizer(optimizer)

        parallel_model = self.multi_gpu_model(model)
        parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model, parallel_model
