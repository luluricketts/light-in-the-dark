import os
import argparse
# import mmcv
import yaml
from yaml import safe_load as yload

from src import CycleGAN, get_data_loader



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
if __name__ == "__main__":
    args = parser.parse_args()
    # cfg = mmcv.Config.fromfile(args.config)
    with open(args.config, 'r') as file:
        cfg = yload(file)


    cfg.out_dir = os.path.join('logs', cfg.exp_name)
    os.makedirs(cfg.out_dir, exist_ok=True)
    # cfg.dump(os.path.join(cfg.out_dir, 'config.py'))
    with open(os.path.join(cfg.out_dir, 'config.py'), 'w') as file:
        yaml.dump(file)
    
    # get data
    dataloader_X_tr = get_data_loader(cfg.data_train, 'high')
    dataloader_Y_tr = get_data_loader(cfg.data_train, 'low')
    dataloader_X_test = get_data_loader(cfg.data_test, 'high')
    dataloader_Y_test = get_data_loader(cfg.data_test, 'low')

    # define model
    cyclegan = CycleGAN(cfg)

    cyclegan.train(dataloader_X_tr, dataloader_Y_tr, cfg.train)