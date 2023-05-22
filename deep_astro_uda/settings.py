import os 
# Default output paths.

DEFAULT_CONFIG_PATH = f"{os.getcwd()}/deepastro_files/config_env/"

# CONFIG OPTIONS

DATASET_NAMES = ["astro-nn", "office_home", "galaxy_zoo", "deep_adversaries", "data"]

# Template configuration dictionaries.

ASTRO_NN_CONFIG = {
    "data": {},
    "model": {},
    "training": {},
    "inference": {},
    "output": {}
}

OFFICE_HOME_CONFIG = {}
GALAXY_ZOO_CONFIG = {}
DEEP_ADVERSARIES_CONFIG = {}
DEFAULT_CONFIG = {}
