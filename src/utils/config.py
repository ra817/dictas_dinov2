import yaml
import torch
import random
import numpy as np
from easydict import EasyDict as edict

def load_config(path="config/config.yaml"):
    """Load YAML config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)

def set_seed(seed=42):
    """Set global random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
