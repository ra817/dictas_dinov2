import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict


MERGE_THRESHOLD = 0.80          # patch must be ≥80% similar to merge
EMA_DECAY = 0.97                # smooth dictionary updates
MAX_DICT_SIZE = 200             # optional limit (adjust as needed)


def train(model, train_loader, val_loader, optimizer, num_epochs, save_dir, patience=5):

    print("\nStarting Training: Warm-Up + Dynamic Dictionary Clustering...\n")

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        #RESET COUNTERS
        key_update_counter = defaultdict(int)
        epoch_merge_count = 0
        epoch_insert_count = 0

        #TRAIN
        model.train()
        running_train_loss = 0.0

        # WARM-UP (projection layers only)
        warmup = (epoch < 5)

        print(f"\nEpoch {epoch} | Warmup={warmup} | Threshold={MERGE_THRESHOLD} | EMA={EMA_DECAY}")

        for imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = imgs.cuda()

            #Forward pass → dynamic lookup
            total_loss, L_recon, L_align, L_smooth, new_keys, new_vals, best_sim, best_idx = model(imgs)
            print(f"total loss:{total_loss}")
            print("DEBUG -- total_loss:", total_loss)
            print("requires_grad:", total_loss.requires_grad)
            print("grad_fn:", total_loss.grad_fn)

            #Train projection layers
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        
            #WARM-UP: SKIP DICT UPDATE
            if warmup:
                running_train_loss += total_loss.item()
                continue


            #DICTIONARY UPDATE
            with torch.no_grad():

                # CASE 1: dictionary empty → insert all keys
                if best_sim is None:
                    for i in range(new_keys.shape[0]):
                        model.dictionary.insert(new_keys[i], new_vals[i])
                        epoch_insert_count += 1
                    continue

                # CASE 2: dictionary exists → merge or insert
                for i in range(new_keys.shape[0]):

                    if best_sim[i] >= MERGE_THRESHOLD:
                        # MERGE using EMA
                        idx = best_idx[i]
                        key_update_counter[int(idx)] += 1
                        epoch_merge_count += 1

                        model.dictionary.keys[idx] = (
                            EMA_DECAY * model.dictionary.keys[idx] +
                            (1 - EMA_DECAY) * new_keys[i]
                        )

                        # normalize
                        model.dictionary.keys[idx] = F.normalize(
                            model.dictionary.keys[idx], dim=-1
                        )

                    else:
                        # INSERT new key if dictionary not full
                        if model.dictionary.keys.shape[0] < MAX_DICT_SIZE:
                            model.dictionary.insert(new_keys[i])
                            epoch_insert_count += 1

            running_train_loss += total_loss.item()

        #END OF TRAIN LOOP

        avg_train_loss = running_train_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f}")
        print(f"Merged: {epoch_merge_count} | Inserted: {epoch_insert_count}")
        print(f"Dict Size: {model.dictionary.keys.shape[0]}")

        #VALIDATION
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.cuda()
                total_loss, *_ = model(imgs)
                running_val_loss += total_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Val Loss: {avg_val_loss:.6f}")

        #EARLY STOP
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save({"state_dict": model.state_dict(),
                        "dict_size": model.dictionary.keys.shape[0]}, save_path)

            print(f"→ Saved BEST model | dict_size={model.dictionary.keys.shape[0]}")

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} | Best Val: {best_val_loss:.6f}")
                break


    # Save last model
    save_path = os.path.join(save_dir, "last_model.pth")
    torch.save({"state_dict": model.state_dict(),
                "dict_size": model.dictionary.keys.shape[0]}, save_path)

    print(f"\nTraining Complete → Saved LAST model | dict_size={model.dictionary.keys.shape[0]}")
