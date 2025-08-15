import os

import sys

import argparse

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Add the project root path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess, split_data_non_iid_with_sizes

from core.federated_training import federated_training

from utils.plot import plot_accuracy_loss

from utils.model import load_global_model,create_cnn_model

from utils.attacks import prepare_mia_data,evaluate_mia

from configs import config

# === cLI Arguments  ===

parser = argparse.ArgumentParser(description="Run Federated Learning with FedAvg.")

parser.add_argument('--dataset', type=str, choices=['pathmnist', 'organamnist', 'bloodmnist'], default='pathmnist', help='Dataset to use')

parser.add_argument('--rounds', type=int, default=10, help='Precise Number of rounds used for model conception')

parser.add_argument('--transform', type=str, choices=['uniform', 'normal'], default='normal', help='Type of Transformation')

parser.add_argument('--noise', type=str, choices=['laplace', 'gaussian'], default='gaussian', help='Type of Noise')

parser.add_argument('--epsilon',  type=float, default=3.0, help='Epsilon value')

parser.add_argument('--samples', type=int, default=1000, help='Precise Number of rounds used for model conception')


args = parser.parse_args()

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


# === Prepare_mia_data ===

mia_data = prepare_mia_data(

    (train_images, train_labels),

    (test_images, test_labels),

    num_samples= args.samples
)

global_model_path=f"results/models/ldpp-avg/final_model_for_dataset_{args.dataset}_num_rounds_{args.rounds}_{args.transform}_{args.noise}_{args.epsilon}.keras"
 
global_model = load_global_model(global_model_path)

csv_dir = f"results/attacks/mia/ldpp-avg/{args.dataset}_num_rounds_{args.rounds}_{args.transform}_{args.noise}_{args.epsilon}/test.csv"

metrics_df = evaluate_mia(global_model, mia_data,output_csv=csv_dir)
