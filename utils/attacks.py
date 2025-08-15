
import numpy as np

import pandas as pd

import os

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def prepare_mia_data(train_data, test_data, num_samples):

    member_indices = np.random.choice(len(train_data[0]), num_samples, replace=False)

    non_member_indices = np.random.choice(len(test_data[0]), num_samples, replace=False)
    
    return {

        "member_images": train_data[0][member_indices],

        "member_labels": train_data[1][member_indices],  # Maintenant des entiers

        "non_member_images": test_data[0][non_member_indices],

        "non_member_labels": test_data[1][non_member_indices]  # Maintenant des entiers
    }



def compute_mia_scores(model, mia_data):

    # Les étiquettes sont maintenant des entiers

    member_probs = model.predict(mia_data["member_images"], verbose=0)

    non_member_probs = model.predict(mia_data["non_member_images"], verbose=0)
    
    # Conversion explicite en entiers (sécurité supplémentaire)

    member_labels = mia_data["member_labels"].astype(int)

    non_member_labels = mia_data["non_member_labels"].astype(int)
    
    member_losses = -np.log(np.take_along_axis(member_probs, member_labels[:, None], axis=1))

    non_member_losses = -np.log(np.take_along_axis(non_member_probs, non_member_labels[:, None], axis=1))
    
    return member_losses.flatten(), non_member_losses.flatten()


def evaluate_mia(global_model, mia_data, output_csv):

    model_path = os.path.join(output_csv)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)


    # Compute member and non-member losses

    member_losses, non_member_losses = compute_mia_scores(global_model, mia_data)

    # Concatenate all losses and labels
    
    all_losses = np.concatenate([member_losses, non_member_losses])
    
    all_labels = np.concatenate([
    
        np.ones_like(member_losses),  # members = 1
    
        np.zeros_like(non_member_losses)  # non-members = 0
    ])

    # Compute AUC-ROC
    
    mia_auc = roc_auc_score(all_labels, -all_losses)

    # Determine optimal threshold using Youden's J statistic
    
    fpr, tpr, thresholds = roc_curve(all_labels, -all_losses)
    
    youden_j = tpr - fpr
    
    optimal_idx = np.argmax(youden_j)
    
    optimal_threshold = thresholds[optimal_idx]

    # Binary predictions (1 = member, 0 = non-member)
    
    binary_predictions = (all_losses < optimal_threshold).astype(int)

    # Accuracy calculation
    
    mia_accuracy = accuracy_score(all_labels, binary_predictions)

    # Store metrics
    
    metrics = [{
    
        'mia_accuracy': mia_accuracy,
    
        'mia_auc': mia_auc
    
    }]
    
    metrics_df = pd.DataFrame(metrics)

    # Save metrics to CSV
    
    metrics_df.to_csv(output_csv, index=False)

    # Print results
    
    print("\nRésultats de l'attaque MIA:")

    print(f"Attack Accuracy: {mia_accuracy:.4f}")
    
    print(f"- AUC-ROC: {mia_auc:.4f}")
    
    print(f"- Average Loss Members : {np.mean(member_losses):.4f}")
    
    print(f"- Average Loss Non-Members : {np.mean(non_member_losses):.4f}")
    
    print(f"- Relative Difference: {(np.mean(non_member_losses) - np.mean(member_losses))/np.mean(member_losses):.2%}")

    return metrics_df

def flip_labels_for_adversarial_clients(client_data, num_adversarial, num_classes):
    
    """
    Invert labels for a given number of adversarial clients using cyclic rotation.
    
    Parameters
    ----------
    client_data : dict or list
        Mapping from client index to (images, labels).
    num_adversarial : int
        Number of adversarial clients to modify (starting from index 0).
    num_classes : int
        Total number of classes in the dataset.

    Returns
    -------
    client_data : same type as input
        Updated client data with flipped labels for adversarial clients.
    """
    for client_idx in range(num_adversarial):
    
        images, labels = client_data[client_idx]
        
        # Cyclic label rotation: 0→1→2→...→0
    
        flipped_labels = (labels + 1) % num_classes
        
        client_data[client_idx] = (images, flipped_labels)
    
    return client_data

