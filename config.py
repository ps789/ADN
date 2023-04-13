from typing import Optional, List
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool
from tensorfn.config import (
    MainConfig,
    Config,
    Optimizer,
    Scheduler,
    DataLoader,
    Instance,
)

class Dataset(Config):
    name: StrictStr
    path: StrictStr
    resolution: StrictInt

class Diffusion(Config):
    beta_schedule: Instance

class Training(Config):
    n_iter: StrictInt
    optimizer: Optimizer
    scheduler: Optional[Scheduler]
    dataloader: DataLoader

class Eval(Config):
    wandb: StrictBool
    save_every: StrictInt
    valid_every: StrictInt
    log_every: StrictInt
    
class Discriminator(Config):
    num_channels: StrictInt
    discriminator_features: StrictInt
    cpu: StrictBool

class DiffusionConfig(MainConfig):
    dataset: Dataset
    generator: Instance
    diffusion: Diffusion
    training: Training
    evaluate: Eval
    discriminator: Discriminator
