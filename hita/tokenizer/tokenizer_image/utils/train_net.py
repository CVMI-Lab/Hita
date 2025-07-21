import os.path as osp
from omegaconf import OmegaConf
from engine.util import instantiate_from_config

def train_ARmodel():

    cfg_file = 'configs/AR_model_config.yaml'
    assert osp.exists(cfg_file)
    config = OmegaConf.load(cfg_file)
    trainer = instantiate_from_config(config.trainer)
    trainer.train()

if __name__ == '__main__':

    train_ARmodel()
