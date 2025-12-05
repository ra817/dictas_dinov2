import os
import matplotlib.pyplot as plt


def plot_loss(train_curve, val_curve, save_dir, filename="loss_curve.png"):
    """
    Plots training & validation loss curves.
    
    Args:
        train_curve (list): List of train loss values per epoch
        val_curve (list): List of validation loss values per epoch
        save_dir (str): Directory where plot image is saved
        filename (str): Output filename for plot
    """

    plt.figure(figsize=(7, 5))
    plt.plot(
        range(1, len(train_curve) + 1),
        train_curve,
        label="Train Loss",
        linewidth=2
    )
    plt.plot(
        range(1, len(val_curve) + 1),
        val_curve,
        label="Val Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    plt.savefig(file_path)
    plt.close()

    print(f"[Plotting] Saved loss curve to: {file_path}")
