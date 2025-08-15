
# configs/config.py

"""

Global configuration file for federated learning experiments.
This file centralizes dataset paths, default parameters, and hospital sizes.

"""

import os

# =========================
# DATA CONFIGURATION
# =========================

# Root data directory

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Supported datasets

DATASETS = ["pathmnist", "organamnist", "bloodmnist"]

# Hospital sizes per dataset (order matches number of clients in main code)

HOSPITAL_SIZES = {

    "pathmnist": [6373, 8625, 6040, 8959, 6640, 8360, 6206, 8794, 6370, 8630, 6405, 8594],

    "organamnist": [2779, 2983, 2446, 3317, 2612, 3152, 2776, 2988, 2811, 2953, 2842, 2902],

    "bloodmnist": [894, 1098, 817, 1175, 904, 1089, 983, 1011, 991, 1103, 926, 968]
}

# =========================
# TRAINING CONFIGURATION
# =========================

# Default number of communication rounds

DEFAULT_ROUNDS = 2

DEFAULT_NUM_CLTS = 12

# Default batch size for local training

BATCH_SIZE = 64

ADVERSARAIAL_RATIO = 0.3

# =========================
# UTILITY FUNCTIONS
# =========================

def get_dataset_path(dataset_name: str) -> str:

    """
    Returns the full path to the dataset file (.npz) for the given dataset name.
    """

    if dataset_name not in DATASETS:

        raise ValueError(f"Dataset '{dataset_name}' is not supported. Choose from {DATASETS}.")

    return os.path.join(DATA_DIR, f"{dataset_name}.npz")


def get_hospital_sizes(dataset_name: str) -> list:

    """
    Returns the list of hospital sizes for the given dataset.
    """
    if dataset_name not in HOSPITAL_SIZES:

        raise ValueError(f"No hospital size configuration found for '{dataset_name}'.")

    return HOSPITAL_SIZES[dataset_name]
