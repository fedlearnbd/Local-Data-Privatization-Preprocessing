
import os

import sys

import argparse

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Add the project root path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess, split_data_non_iid_with_sizes

from core.federated_training import federated_training_with_ldpp

from utils.plot import plot_accuracy_loss

from utils.attacks import flip_labels_for_adversarial_clients

from configs import config

# === cLI Arguments  ===

parser = argparse.ArgumentParser(description="Run Federated Learning with Ldpp-krum.")

parser.add_argument('--dataset', type=str, choices=['pathmnist', 'organamnist', 'bloodmnist'], default='pathmnist', help='Dataset to use')

parser.add_argument('--rounds', type=int, default=150, help='Number of communication rounds')

parser.add_argument('--attacker_ratio', type=float, default=0.3, help='Adversarial Attackers Ratio')

parser.add_argument('--transform', type=str, choices=['uniform', 'normal'], default='normal', help='Type of Transformation')

parser.add_argument('--noise', type=str, choices=['laplace', 'gaussian'], default='gaussian', help='Type of Noise')

parser.add_argument('--epsilon',  type=float, default=3.0, help='Epsilon value')

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


input_shape = train_images.shape[1:]

num_classes = len(set(train_labels))


# === Training ===

global_model, metrics_history, round_times, losses, accuracies = federated_training_with_ldpp(

    train_images, train_labels,

    test_images, test_labels,

    client_sizes=hospital_sizes,

    input_shape=input_shape,

    num_classes=num_classes,

    num_rounds=args.rounds,

    transform_type=args.transform,

    noise_type=args.noise,

    epsilon=args.epsilon,
        
    aggregation_method="krum",  # "fedavg" or "krum"
    
    num_adversarial=int(args.attacker_ratio * config.DEFAULT_NUM_CLTS) # We have 12 clients

)

# === Model backup ===

model_path = f'results/models/ldpp-krum/lfa_attacks/final_model_dataset_{args.dataset}_num_rounds_{args.rounds}_advers_ratio_{args.attacker_ratio}.keras'

os.makedirs(os.path.dirname(model_path), exist_ok=True)

global_model.save(model_path)

# === Saving metrics ===

metrics_df = pd.DataFrame(metrics_history)

metric_path = f'results/metrics/ldpp-krum/lfa_attacks/metrics_{args.dataset}_rnds_{args.rounds}_ratio_{args.attacker_ratio}.csv'

os.makedirs(os.path.dirname(metric_path), exist_ok=True)

metrics_df.to_csv(metric_path, index=False)

# === Trend chart ===

graph_df = pd.DataFrame(
    {
    
    'round': list(range(1, len(losses)+1)),
    
    'loss': losses,
    
    'accuracy': accuracies,
    
    'time': round_times
    }

    )

graph_path=f'results/metrics/ldpp-krum/lfa_attacks/graph_{args.dataset}_rnds_{args.rounds}_ratio_{args.attacker_ratio}.csv'

os.makedirs(os.path.dirname(graph_path), exist_ok=True)

graph_df.to_csv(graph_path, index=False)

# === Plot ===

plot_accuracy_loss(graph_df,graph_path)


