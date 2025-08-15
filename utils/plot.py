import matplotlib.pyplot as plt

def plot_accuracy_loss(graph_df, path):
    """
    Plot accuracy and loss evolution over rounds.
    
    Parameters:
    -----------
    graph_df : pandas.DataFrame
        DataFrame containing columns: 'round', 'accuracy', 'loss'.
    path : str
        Path (without extension) to save the figure as PNG.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(graph_df['round'], graph_df['accuracy'], marker='o')
    plt.title("Évolution de l'Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(graph_df['round'], graph_df['loss'], marker='o', color='red')
    plt.title("Évolution de la Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()

    # Save figure first
    plt.savefig(path + ".png")
    # Then optionally display
    plt.show()
