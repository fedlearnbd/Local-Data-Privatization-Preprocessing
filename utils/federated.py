import numpy as np 

from tensorflow.keras.callbacks import EarlyStopping

from scipy.spatial.distance import cdist


def federated_averaging(models):

    avg_weights = []

    for weights in zip(*[model.get_weights() for model in models]):

        avg_weights.append(np.mean(weights, axis=0))

    return avg_weights

"""
def train_local_model(model, train_images, train_labels, epochs=1, batch_size=32):

    #print(f"Training on images shape: {train_images.shape}, labels shape: {train_labels.shape}")

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=epochs, verbose=0)
"""
def train_local_model(model, train_images, train_labels, epochs=1, batch_size=32):
    # Définition du callback EarlyStopping
    early_stop = EarlyStopping(
        monitor='loss',            # Tu peux mettre 'val_loss' si tu fais validation_split
        patience=3,                # Stoppe si la perte ne diminue pas après 3 epochs
        restore_best_weights=True # Restaure les meilleurs poids trouvés
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )



def krum_aggregation(models, f=1):

    weights_list = [np.concatenate([w.flatten() for w in model.get_weights()]) for model in models]
    n = len(weights_list)
    scores = []

    # Compute pairwise distances between all models
    distance_matrix = cdist(weights_list, weights_list, metric='euclidean')

    for i in range(n):
        distances = np.sort(distance_matrix[i])[1:n - f]  # exclude self and take n - f - 1 closest
        score = np.sum(distances)
        scores.append(score)

    # Select the model with the lowest score
    selected_index = np.argmin(scores)
    return models[selected_index].get_weights()
