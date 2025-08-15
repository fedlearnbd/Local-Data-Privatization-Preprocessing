
import tensorflow as tf

import os

from tensorflow.keras.models import load_model

def create_cnn_model(input_shape, num_classes, verbose=True):

    model = tf.keras.models.Sequential([
    
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
        tf.keras.layers.Flatten(),
    
        tf.keras.layers.Dense(128, activation='relu'),
    
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def load_global_model(model_path):
    """
    Load a trained global model from the given path.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file (.h5).

    Returns
    -------
    model : tensorflow.keras.Model
        Loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the given path.
    """
    if not os.path.exists(model_path):
    
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")

    return load_model(model_path)
