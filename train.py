import argparse
import os
import shutil

import tensorflow as tf
from dotmap import DotMap
from keras import backend as K

from base.base_data_loader import BaseDataLoader
from data_loader.mnist_data_loader import MNISTDataLoader
from utils.config import process_config
from model_trainer_builder import build_model_and_trainer

def main(use_horovod: bool, gpus: int, checkpoint: int, config_path: str) -> None:
    config = process_config(config_path, use_horovod, gpus, checkpoint)

    # create tensorflow session and set as keras backed
    tf_config = tf.ConfigProto()

    if config.trainer.use_horovod:
        import horovod.keras as hvd

        hvd.init()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.visible_device_list = str(hvd.local_rank())

    is_master = not config.trainer.use_horovod
    if not is_master:
        import horovod.keras as hvd

        is_master = hvd.rank() == 0

    if is_master and not os.path.exists(config.exp.source_dir):
        # copy source files
        shutil.copytree(
            os.path.abspath(os.path.curdir),
            config.exp.source_dir,
            ignore=lambda src, names: {"datasets", "__pycache__", ".git", "experiments", "venv"})

    tf_sess = tf.Session(config=tf_config)
    K.set_session(tf_sess)
    data_loader = MNISTDataLoader(config=config)

    _, trainer = build_model_and_trainer(config, data_loader)

    print(f"Start Training Experiment {config.exp.name}")
    trainer.train()

if __name__ == '__main__':
    print(os.path.abspath(os.curdir), os.getpid())

    ap = argparse.ArgumentParser()
    # training env
    ap.add_argument("--horovod", action="store_true", help="use horovod")
    ap.add_argument("--gpus", type=int, default=1, help="number of gpus to use if horovod is disabled")

    # training config
    ap.add_argument("--config", type=str, default="configs/config.yml", help="config path to use")
    ap.add_argument("--chkpt", type=int, default=0, help="checkpoint to continue")

    args = vars(ap.parse_args())

    main(args["horovod"], args["gpus"], args["chkpt"], args["config"])