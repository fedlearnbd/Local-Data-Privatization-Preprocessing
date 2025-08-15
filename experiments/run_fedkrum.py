
# main.py

import os

import sys

import argparse

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import warnings

import logging

# Suppress Python warnings

warnings.filterwarnings('ignore')

# Add the project root path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess

from core.federated_training import federated_training

from utils.plot import plot_accuracy_loss

from configs import config

MODEL_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "models", "krum"))

METRICS_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "metrics", "krum"))

PLOTS_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "plots", "krum"))

# -----------------------------
# Deleting Logs
# -----------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=all, 1=info, 2=warning, 3=error

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Désactive le message oneDNN

# Supprimer les warnings Python

warnings.filterwarnings('ignore')

# Désactiver les logs internes de TensorFlow

logging.getLogger("tensorflow").setLevel(logging.FATAL)

# Option pour TF 1.x shims (compatibilité)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# === CLI Arguments ===

parser = argparse.ArgumentParser(description="Run Federated Learning with Krum.")

parser.add_argument(

    '--dataset',

    type=str,

    choices=config.DATASETS,

    default='pathmnist',

    help='Dataset to use'
)

parser.add_argument(

    '--rounds',

    type=int,

    default=config.DEFAULT_ROUNDS,

    help='Number of communication rounds'
)

args = parser.parse_args()

# === Introduction ===

print("=" * 60)

print(f"Federated Learning with Krum Aggregation")

print(f"Dataset: {args.dataset}")

print(f"Rounds: {args.rounds}")

print("=" * 60)

# === Configurations ===

file_path = config.get_dataset_path(args.dataset)

hospital_sizes = config.get_hospital_sizes(args.dataset)

# === Data loading and pre-processing ===

train_images, train_labels, test_images, test_labels = preprocess(file_path)

# Dataset-specific preprocessing

if args.dataset == 'organamnist':

    train_images = np.expand_dims(train_images, axis=-1)

    test_images = np.expand_dims(test_images, axis=-1)

# Convert one-hot labels to integers if necessary

if train_labels.ndim > 1:

    train_labels = train_labels.argmax(axis=-1)

if test_labels.ndim > 1:

    test_labels = test_labels.argmax(axis=-1)


input_shape = train_images.shape[1:]

num_classes = len(set(train_labels))

# === Training ===

global_model, metrics_history, round_times, losses, accuracies = federated_training(

    train_images, train_labels,

    test_images, test_labels,

    client_sizes=hospital_sizes,

    input_shape=input_shape,

    num_classes=num_classes,

    num_rounds=args.rounds,
    
    aggregation='krum'
)

# === Model backup ===

model_path = os.path.join(MODEL_SAVE_DIR, f'final_model_for_dataset_{args.dataset}_num_rounds_{args.rounds}.keras')

os.makedirs(os.path.dirname(model_path), exist_ok=True)

global_model.save(model_path)

# === Saving metrics ===

metrics_df = pd.DataFrame(metrics_history)

metric_path = os.path.join(METRICS_SAVE_DIR, f'metrics_{args.dataset}_{args.rounds}.csv')

os.makedirs(os.path.dirname(metric_path), exist_ok=True)

metrics_df.to_csv(metric_path, index=False)

# === Trend chart data ===

graph_df = pd.DataFrame({

    'round': list(range(1, len(losses) + 1)),

    'loss': losses,

    'accuracy': accuracies,

    'time': round_times

})

graph_path = os.path.join(PLOTS_SAVE_DIR, f'graph_{args.dataset}_{args.rounds}.csv')

os.makedirs(os.path.dirname(graph_path), exist_ok=True)

pathplot = os.path.join(PLOTS_SAVE_DIR, f'graph_for_{args.dataset}_numrds_{args.rounds}')

os.makedirs(os.path.dirname(pathplot), exist_ok=True)

graph_df.to_csv(graph_path, index=False)

# Plot and save

plot_accuracy_loss(graph_df, pathplot)
