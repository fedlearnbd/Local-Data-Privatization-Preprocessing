
from tensorflow.keras.utils import to_categorical

import numpy as np

import tensorflow as tf

import os

from sklearn.preprocessing import QuantileTransformer

from scipy.stats import laplace, norm

from collections import defaultdict

def normalize_images(images):

    """Normalize image pixel values to the range [0, 1]."""

    return images.astype('float32') / 255.0

def resize_images(images, target_size=(32, 32)):

    return tf.image.resize(images, target_size).numpy()

def preprocess(file_path):

    """Load and preprocess data from the given file path."""
    
    if not os.path.exists(file_path):
    
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    # Load the dataset from the .npz file
    
    data = np.load(file_path)
    
    # Check if required keys are present in the loaded data
    
    if not all(key in data for key in ['train_images', 'train_labels', 'test_images', 'test_labels']):
    
        raise KeyError("The dataset must contain 'train_images', 'train_labels', 'test_images', and 'test_labels'.")
    
    train_images = data['train_images']
    
    train_labels = data['train_labels']
    
    test_images = data['test_images']
    
    test_labels = data['test_labels']
    
    # Normalize images
    
    train_images = normalize_images(train_images)
    
    test_images = normalize_images(test_images)
    
    # Ensure labels are one-hot encoded
    
    num_classes = len(np.unique(train_labels))
    
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    
    test_labels = to_categorical(test_labels, num_classes=num_classes)
    
    return train_images, train_labels, test_images, test_labels

def load_dataset(dataset_name, data_dir='../data/'):

    file_map ={

        'pathmnist': 'pathmnist.npz',

        'organamnist': 'organamnist.npz',

        'bloodmnist': 'bloodmnist.npz',
    }

    if dataset_name not in file_map:

        raise ValueError(f"Dataset '{dataset_name}' non supporté.")

    file_path = os.path.join(data_dir, file_map[dataset_name])

    return preprocess(file_path)


def split_data_non_iid_with_sizes(x_data, y_data, num_clients, client_sizes, num_classes, seed=None):
    """
    Répartition non-IID avec tailles spécifiques par client.

    Arguments :
    - x_data : données d'entrée (np.array ou liste)
    - y_data : labels (np.array ou liste)
    - num_clients : nombre de clients
    - client_sizes : liste contenant la taille des données pour chaque client
    - num_classes : nombre total de classes dans le dataset
    - seed : pour la reproductibilité

    Retour :
    - dict {client_id: (x_client, y_client)}
    """
    assert len(client_sizes) == num_clients, "La taille de client_sizes doit être égale à num_clients."
    assert sum(client_sizes) <= len(x_data), "La somme des tailles dépasse le nombre total d'échantillons."

    if seed is not None:
        np.random.seed(seed)

    # Regrouper les indices des données par classe
    class_indices = defaultdict(list)
    for idx, label in enumerate(y_data):
        class_indices[label].append(idx)

    # Mélanger les indices pour chaque classe
    for indices in class_indices.values():
        np.random.shuffle(indices)

    client_data = {i: ([], []) for i in range(num_clients)}

    current_class = 0
    class_cycle = list(range(num_classes))

    for client_id, size in enumerate(client_sizes):
        assigned = 0
        while assigned < size:
            class_label = class_cycle[current_class % num_classes]
            available_indices = class_indices[class_label]

            if available_indices:
                index = available_indices.pop()
                client_data[client_id][0].append(x_data[index])
                client_data[client_id][1].append(y_data[index])
                assigned += 1

            current_class += 1

    # Convertir les listes en numpy arrays
    for client_id in client_data:
        x, y = client_data[client_id]
        client_data[client_id] = (np.array(x), np.array(y))

    return client_data


def ldpp_transform(images, transform_type, noise_type, epsilon):
    if transform_type not in ['normal', 'uniform']:
        raise ValueError(f"Transform type '{transform_type}' non supporté. Utiliser 'normal' ou 'uniform'.")

    original_shape = images.shape
    images_flat = images.reshape(-1, 1)

    qt = QuantileTransformer(output_distribution=transform_type)
    transformed = qt.fit_transform(images_flat)

    sensitivity = 1.0

    if noise_type == 'laplace':
        scale = sensitivity / epsilon
        noise = laplace.rvs(scale=scale, size=transformed.shape)
    elif noise_type == 'gaussian':
        sigma = np.sqrt(2 * np.log(1.25)) * sensitivity / epsilon
        noise = norm.rvs(scale=sigma, size=transformed.shape)
    else:
        raise ValueError(f"Type de bruit '{noise_type}' non reconnu. Utiliser 'laplace' ou 'gaussian'.")

    privatized = transformed + noise
    reversed_data = qt.inverse_transform(privatized)

    return reversed_data.reshape(original_shape)


def direct_noise_injection(images, noise_type, epsilon):

    original_shape = images.shape
    images_flat = images.reshape(-1, 1)


    sensitivity = 1.0

    if noise_type == 'laplace':
        scale = sensitivity / epsilon
        noise = laplace.rvs(scale=scale, size=images.shape)
    elif noise_type == 'gaussian':
        sigma = np.sqrt(2 * np.log(1.25)) * sensitivity / epsilon
        noise = norm.rvs(scale=sigma, size=images.shape)
    else:
        raise ValueError(f"Type de bruit '{noise_type}' non reconnu. Utiliser 'laplace' ou 'gaussian'.")

    privatized = images + noise
    

    return privatized.reshape(original_shape)

