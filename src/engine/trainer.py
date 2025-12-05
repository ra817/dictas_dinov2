import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.plot import plot_loss


def train(model, train_loader, val_loader, optimizer, num_epochs, save_dir, patience=5):

    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    print("Starting DictA-S Training with Validation...")

    for epoch in range(num_epochs):

        #TRAINING LOOP
        model.train()
        running_train_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for imgs in progress:
            imgs = imgs.cuda()

            #Forward pass
            total_loss, L_recon, L_align, L_smooth = model(imgs)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_train_loss += total_loss.item()

            progress.set_postfix({
                "L_recon": f"{L_recon.item():.4f}",
                "L_align": f"{L_align.item():.4f}",
                "L_smooth": f"{L_smooth.item():.4f}"
            })

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f}")

        #VALIDATION LOOP
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.cuda()

                total_loss, _, _, _ = model(imgs)
                running_val_loss += total_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1:03d} | Val Loss:   {avg_val_loss:.6f}")


        #EARLY STOPPING
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({"state_dict": model.state_dict(),
                "global_dict_size": model.dictionary.keys.shape[0],
            }, best_model_path)
            print(f"current dict_size is:{model.dictionary.keys.shape[0]}")
            print(f"→ Saved BEST model (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                break


    #Save final model
    last_model_path = os.path.join(save_dir, "last_model.pth")
    torch.save({"state_dict": model.state_dict(),
        "global_dict_size": model.dictionary.keys.shape[0],
    }, last_model_path)

    print(f"Saved LAST model to: {last_model_path}")

    #Plot curves
    plot_loss(train_losses, val_losses, save_dir)

    