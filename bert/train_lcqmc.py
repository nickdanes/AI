import train
from args import Config

config = Config()
train.main(config.lcqmc_train_path, config.lcqmc_dev_path, config.lcqmc_model_path)
