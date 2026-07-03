import os
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
CONFIG_DIR = os.path.join(ROOT_DIR, "assets/configs")
CONFIGURATION_FILE = os.path.join(CONFIG_DIR, "app_config.yaml")


def load_config(path=None):
    import yaml
    from easydict import EasyDict as edict
    p = path or CONFIGURATION_FILE
    with open(p, "r") as f:
        return edict(yaml.safe_load(f))


app_profile = load_config()