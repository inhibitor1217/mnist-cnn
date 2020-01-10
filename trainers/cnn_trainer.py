import datetime
import os
from collections import defaultdict
from typing import Optional

import numpy as np
from PIL import Image
from keras import backend as K, Model
from keras.callbacks import LearningRateScheduler

from base.base_data_loader import BaseDataLoader
from base.base_trainer import BaseTrainer
from utils.callback import ScalarCollageTensorBoard, ModelCheckpointWithKeepFreq, OptimizerSaver, ModelSaver, \
    TrainProgressAlertCallback

class CNNTrainer(BaseTrainer):
    def __init__(self, model, parallel_model, data_loader, config):
        super().__init__(data_loader, config)
        self.serial_model = model
        self.model = parallel_model

        self.model_callbacks = defaultdict(list)
        self.init_callbacks()

    def init_callbacks(self) -> None:
        if self.config.trainer.use_lr_decay:
            # linear decay from the half of max_epochs
            def lr_scheduler(lr, epoch, max_epochs):
                return min(lr, 2 * lr * (1 - epoch / max_epochs))

            self.model_callbacks['model'].append(
                LearningRateScheduler(schedule=lambda epoch: lr_scheduler(self.config.model.generator.lr, epoch,
                                                                          self.config.trainer.num_epochs)))

        # if horovod used, only worker 0 saves checkpoints
        is_master = True
        is_local_master = True
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd

            is_master = hvd.rank() == 0
            is_local_master = hvd.local_rank() == 0

        # horovod callbacks
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd

            self.model_callbacks["model"].append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            self.model_callbacks["model"].append(hvd.callbacks.MetricAverageCallback())
            self.model_callbacks["model"].append(
                hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))

        if is_local_master:
            # model saver
            self.model_callbacks["serial_model"].append(
                ModelCheckpointWithKeepFreq(
                    filepath=os.path.join(self.config.exp.checkpoints_dir, "{epoch:04d}-combined.hdf5"),
                    keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                    save_checkpoint_freq=self.config.trainer.save_checkpoint_freq,
                    save_best_only=False,
                    save_weights_only=True,
                    verbose=1))

            # save optimizer weights
            for model_name in ['model']:
                self.model_callbacks[model_name].append(OptimizerSaver(self.config, model_name))
        if is_master:
            # save individual models
            for model_name in ['model']:
                self.model_callbacks[model_name].append(
                    ModelSaver(
                        checkpoint_dir=self.config.exp.checkpoints_dir,
                        keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                        model_name=model_name,
                        num_epochs=self.config.trainer.num_epochs,
                        verbose=1))

            # send notification to telegram channel on train start and end
            self.model_callbacks["model"].append(TrainProgressAlertCallback(experiment_name=self.config.exp.name,
                                                                               total_epochs=self.config.trainer.num_epochs))

            # tensorboard callback
            self.model_callbacks["model"].append(
                ScalarCollageTensorBoard(log_dir=self.config.exp.tensorboard_dir,
                                         batch_size=self.config.trainer.batch_size,
                                         write_images=True))

        # initialize callbacks by setting model and params
        epochs = self.config.trainer.num_epochs
        steps_per_epoch = self.data_loader.get_train_data_size() // self.config.trainer.batch_size
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")

            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.set_model(model)
                callback.set_params({
                    "batch_size": self.config.trainer.batch_size,
                    "epochs": epochs,
                    "steps": steps_per_epoch,
                    "samples": self.data_loader.get_train_data_size(),
                    "verbose": True,
                    "do_validation": False,
                    "model_name": model_name,
                })

    def train(self):
        train_data_gen = self.data_loader.get_train_data_generator()
        batch_size = self.config.trainer.batch_size

        steps_per_epoch = self.data_loader.get_train_data_size() // batch_size
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd
            steps_per_epoch //= hvd.size()
        assert steps_per_epoch > 0

        epochs = self.config.trainer.num_epochs
        start_time = datetime.datetime.now()

        self.on_train_begin()
        for epoch in range(self.config.trainer.epoch_to_continue, epochs):
            self.on_epoch_begin(epoch, {})
            epoch_logs = defaultdict(float)
            for step in range(1, steps_per_epoch + 1):
                batch_logs = {'batch': step, 'size': batch_size}
                self.on_batch_begin(step, batch_logs)

                x, y = next(train_data_gen)

                [loss, accuracy] = self.model.train_on_batch(x, y)

                metric_logs = {
                    'loss': loss,
                    'accuracy': accuracy
                }

                batch_logs.update(metric_logs)
                for metric_name in metric_logs.keys():
                    if metric_name in epoch_logs:
                        epoch_logs[metric_name] += metric_logs[metric_name]
                    else:
                        epoch_logs[metric_name] = metric_logs[metric_name]

                print_str = f"[Epoch {epoch + 1}/{epochs}] [Batch {step}/{steps_per_epoch}]"
                deliminator = ' '
                for metric_name, metric_value in metric_logs.items():
                    if 'accuracy' in metric_name:
                        print_str += f"{deliminator}{metric_name}={metric_value*100:.1f}%"
                    elif 'loss' in metric_name:
                        print_str += f"{deliminator}{metric_name}={metric_value:.4f}"
                    else:
                        print_str += f"{deliminator}{metric_name}={metric_value}"
                    if deliminator == ' ':
                        deliminator = ',\t'

                print_str += f", time: {datetime.datetime.now() - start_time}"
                print(print_str, flush=True)

                self.on_batch_end(step, batch_logs)

            # sum to average
            for k in epoch_logs:
                epoch_logs[k] /= steps_per_epoch
            epoch_logs = dict(epoch_logs)

            # additional log
            epoch_logs['train/lr'] = K.get_value(self.model.optimizer.lr)

            self.on_epoch_end(epoch, epoch_logs)

        self.predict_test_images()
        self.on_train_end()

    def predict_test_images(self) -> None:
        data_generator = self.data_loader.get_test_data_generator()
        data_size = self.data_loader.get_test_data_size()
        
        steps = data_size // self.config.trainer.batch_size
        correct = 0.

        for _ in range(steps):
            x, y = next(data_generator)
            result_raw = self.model.predict(x)
            result = result_raw.argmax(axis=1)
            correct_labels = np.sum(result == y)
            correct += correct_labels

        correct /= data_size
        print(f'Test set accuracy: {correct}')

    def on_batch_begin(self, batch: int, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict]) -> None:
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict]) -> None:
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            model.stop_training = False
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_end(logs)