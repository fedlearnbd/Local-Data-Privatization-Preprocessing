import numpy as np
import os
from scipy.stats import entropy, wasserstein_distance, ks_2samp
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.preprocessing import *

# Configuration GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def calculate_kl_divergence(original, privatized, bins=100):
    """Calcule la divergence KL entre deux distributions"""
    hist_orig, bin_edges = np.histogram(original.flatten(), bins=bins, density=True)
    hist_priv, _ = np.histogram(privatized.flatten(), bins=bin_edges, density=True)
    hist_orig = np.clip(hist_orig, 1e-10, 1)
    hist_priv = np.clip(hist_priv, 1e-10, 1)
    return entropy(hist_orig, hist_priv)

def calculate_wasserstein(original, privatized):
    """Calcule la distance de Wasserstein entre deux ensembles de données"""
    orig_flat = original.flatten()
    priv_flat = privatized.flatten()
    sample_size = min(5000, len(orig_flat))
    rng = np.random.default_rng(42)
    return wasserstein_distance(
        rng.choice(orig_flat, sample_size, replace=False),
        rng.choice(priv_flat, sample_size, replace=False)
    )

def calculate_ks_test(original, privatized):
    """Calcule la statistique KS entre deux distributions"""
    orig_flat = original.flatten()
    priv_flat = privatized.flatten()
    sample_size = min(5000, len(orig_flat))
    rng = np.random.default_rng(42)
    return ks_2samp(
        rng.choice(orig_flat, sample_size, replace=False),
        rng.choice(priv_flat, sample_size, replace=False),
        method='exact'
    ).statistic
