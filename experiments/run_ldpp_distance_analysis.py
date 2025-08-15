
import os

import sys

import argparse

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import ldpp_transform

from configs import config

# === Configuration GPU ===

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:

        print(e)

# === Imports ===

from utils.distances import (

    calculate_wasserstein,

    calculate_kl_divergence,

    calculate_ks_test

)

from utils.preprocessing import preprocess

# === Parser Argument  ===

parser = argparse.ArgumentParser(description="Analysis of distances between original and private data (LDPP)")

parser.add_argument('--dataset', type=str, choices=['pathmnist', 'organamnist', 'bloodmnist'], default='pathmnist', help='Dataset à utiliser')

parser.add_argument('--epsilon', type=float, default=3.0, help="Valeur de l'epsilon pour LDPP")

parser.add_argument('--noise', type=str, choices=['laplace', 'gaussian'], default='gaussian', help="Type de bruit")

parser.add_argument('--transform', type=str, choices=['normal', 'uniform'], default='normal')

parser.add_argument('--samples', type=int, default=60000, help="Nombre d’échantillons à évaluer")

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



# Réduction à un nombre d'échantillons

train_images = train_images[:args.samples]

# === Application Of LDPP ===

print(f"Application of LDPP with ε={args.epsilon}, noise={args.noise}, transform={args.transform} - (Please Wait for Calculation, that's could take a moment)")

privatized_images = ldpp_transform(train_images, args.transform, args.noise, args.epsilon)

# === calculation of distances ===

print("\n Calculation of statistical distances ...")

wd_values, kl_values, ks_values = [], [], []

for orig, priv in zip(train_images, privatized_images):

    wd_values.append(calculate_wasserstein(orig, priv))

    kl_values.append(calculate_kl_divergence(orig, priv))

    ks_values.append(calculate_ks_test(orig, priv))

# === Histogram display ===

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)

plt.hist(wd_values, bins=50, alpha=0.7, color='blue')

plt.title(f'Wasserstein (ϵ={args.epsilon})')

plt.xlabel('Distance')

plt.subplot(1, 3, 2)

plt.hist(kl_values, bins=50, alpha=0.7, color='green')

plt.title(f'KL Divergence (ϵ={args.epsilon})')

plt.xlabel('Divergence')

plt.subplot(1, 3, 3)

plt.hist(ks_values, bins=50, alpha=0.7, color='red')

plt.title(f'KS Statistic (ϵ={args.epsilon})')

plt.xlabel('KS Statistic')

plt.tight_layout()

plt.show()

# === Results Summary ===

print(f"\n--- Results aggregated on {args.samples} images ---")

print(f"Wasserstein : {np.mean(wd_values):.4f} ± {np.std(wd_values):.4f}")

print(f"KL Divergence : {np.mean(kl_values):.4f} ± {np.std(kl_values):.4f}")

print(f"KS Statistic  : {np.mean(ks_values):.4f} ± {np.std(ks_values):.4f}")

