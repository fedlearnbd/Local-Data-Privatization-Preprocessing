# core/federated_training.py

import time

import numpy as np

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score, precision_score

from tensorflow.keras.backend import clear_session

from utils.model import create_cnn_model

from utils.federated import train_local_model, federated_averaging, krum_aggregation

from utils.preprocessing import *

from utils.attacks import flip_labels_for_adversarial_clients

def federated_training(train_images, train_labels,test_images, test_labels,client_sizes,num_clients=12, num_rounds=150,batch_size=64, input_shape=None, num_classes=None,num_adversarial=0,aggregation=None):
    
    start_time = time.time()
    
    client_data = clients_data = split_data_non_iid_with_sizes(train_images, train_labels, num_clients, client_sizes, num_classes, seed=42)

    # 3. Flip labels for adversarial clients (if any)

    if num_adversarial > 0:
    
        client_data = flip_labels_for_adversarial_clients(client_data, num_adversarial, num_classes)
    
        print(f"Label flipping applied to {num_adversarial} adversarial clients.")
    
    global_model = create_cnn_model(input_shape, num_classes)

    global_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # One-hot encoding des labels pour AUC

    test_labels_onehot = to_categorical(test_labels, num_classes)

    metrics_history = []

    graph = []

    losses, accuracies = [], []

    f1_scores, auc_scores = [], []

    precisions, recalls = [], []

    round_times = []

    for round_num in range(num_rounds):

        print(f'\n--- Round {round_num + 1}/{num_rounds} ---')

        local_models = []

        for i, (images, labels) in client_data.items():

            print(f'Client {i + 1}/{num_clients}')

            local_model = create_cnn_model(input_shape, num_classes)

            local_model.set_weights(global_model.get_weights())

            train_local_model(local_model, images, labels)

            local_models.append(local_model)

            clear_session()

        # FedAvg 
        
        if(aggregation=='avg'):
            
            global_weights = federated_averaging(local_models)

        else:
                      
            global_weights = krum_aggregation(local_models)


        global_model.set_weights(global_weights)

        # Evaluation

        exec_time = time.time() - start_time

        round_times.append(exec_time)

        global_loss, global_acc = global_model.evaluate(test_images, test_labels, verbose=0)

        y_pred = global_model.predict(test_images)

        y_pred_classes = np.argmax(y_pred, axis=1)

        f1 = f1_score(test_labels, y_pred_classes, average='weighted')

        auc = roc_auc_score(test_labels, y_pred, multi_class='ovr')

        precision = precision_score(test_labels, y_pred_classes,  average='macro', zero_division=0)
        
        recall = recall_score(test_labels, y_pred_classes, average='weighted')

        print(f"Acc: {global_acc:.4f} | Loss: {global_loss:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")


        metrics_history.append({

            'round': round_num + 1,
            
            'loss': global_loss,
            
            'accuracy': global_acc,
            
            'f1_score': f1,
            
            'auc': auc,
            
            'precision': precision,
            
            'recall': recall,
            
            'time': exec_time
        })

        losses.append(global_loss)

        accuracies.append(global_acc)
        
        f1_scores.append(f1)
        
        auc_scores.append(auc)
        
        precisions.append(precision)
        
        recalls.append(recall)

    return global_model, metrics_history, round_times, losses, accuracies




def federated_training_with_ldpp(
        
    train_images, train_labels,
    
    test_images, test_labels,
    
    client_sizes,
    
    num_clients=12, num_rounds=150,
    
    transform_type=None, noise_type=None, epsilon=None,
    
    batch_size=64, input_shape=None, num_classes=None,
    
    aggregation_method="fedavg",  # "fedavg" or "krum"
    
    num_adversarial=0              # number of adversarial clients for label flipping
):
    """

    Federated training with LDPP preprocessing, optional adversarial clients,
    and selectable aggregation method.
    
    """
    start_time = time.time()

    # 1. Data splitting
    
    client_data = split_data_non_iid_with_sizes(
    
        train_images, train_labels, num_clients, client_sizes, num_classes, seed=42
    )

    # 2. Apply LDPP transformation locally
    
    for i, (images, labels) in client_data.items():
    
        images = ldpp_transform(images, transform_type, noise_type, epsilon)
    
        client_data[i] = (images, labels)
    
        print(f"LDPP applied to client {i+1}")

    # 3. Flip labels for adversarial clients (if any)

    if num_adversarial > 0:
    
        client_data = flip_labels_for_adversarial_clients(client_data, num_adversarial, num_classes)
    
        print(f"Label flipping applied to {num_adversarial} adversarial clients.")

    # 4. Initialize global model
    
    global_model = create_cnn_model(input_shape, num_classes)
    
    global_model.compile(
    
        optimizer='adam',
    
        loss='sparse_categorical_crossentropy',
    
        metrics=['accuracy']
    )

    test_labels_onehot = to_categorical(test_labels, num_classes)

    metrics_history = []
    
    losses, accuracies = [], []
    
    f1_scores, auc_scores = [], []
    
    precisions, recalls = [], []
    
    round_times = []

    # 5. Federated rounds
    
    for round_num in range(num_rounds):
    
        print(f'\n--- Round {round_num + 1}/{num_rounds} ---')
    
        local_models = []

        for i, (images, labels) in client_data.items():

            print(f'Client {i + 1}/{num_clients}')

            local_model = create_cnn_model(input_shape, num_classes)

            local_model.set_weights(global_model.get_weights())

            train_local_model(local_model, images, labels)

            local_models.append(local_model)

            clear_session()

        # 6. Aggregation

        if aggregation_method.lower() == "krum":

            global_weights = krum_aggregation(local_models)

        elif aggregation_method.lower() == "fedavg":

            global_weights = federated_averaging(local_models)

        else:

            raise ValueError("Invalid aggregation method. Choose 'fedavg' or 'krum'.")

        global_model.set_weights(global_weights)

        # 7. Evaluation

        exec_time = time.time() - start_time

        round_times.append(exec_time)

        global_loss, global_acc = global_model.evaluate(test_images, test_labels, verbose=0)

        y_pred = global_model.predict(test_images)

        y_pred_classes = np.argmax(y_pred, axis=1)

        f1 = f1_score(test_labels, y_pred_classes, average='weighted')

        auc = roc_auc_score(test_labels, y_pred, multi_class='ovr')

        precision = precision_score(test_labels, y_pred_classes, average='macro', zero_division=0)

        recall = recall_score(test_labels, y_pred_classes, average='weighted')

        print(f"Acc: {global_acc:.4f} | Loss: {global_loss:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        metrics_history.append({

            'round': round_num + 1,

            'loss': global_loss,

            'accuracy': global_acc,

            'f1_score': f1,

            'auc': auc,

            'precision': precision,

            'recall': recall,

            'time': exec_time
        })

        losses.append(global_loss)

        accuracies.append(global_acc)

        f1_scores.append(f1)

        auc_scores.append(auc)

        precisions.append(precision)

        recalls.append(recall)

    return global_model, metrics_history, round_times, losses, accuracies
