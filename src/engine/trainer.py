import os
import json
import torch
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from src.utils.plot import plot_loss
from src.utils.dict_log import DictLogger
from src.models.dictionary.dict_update import bootstrap_update, normal_update

#thresholds
BOOTSTRAP_MERGE_G = 0.95
BOOTSTRAP_MERGE_P = 0.95
GLOBAL_MERGE_T = 0.5    # global dictionary is more generic → low threshold
PCB_MERGE_T    = 0.7     # specific PCB → higher merge threshold
GLOBAL_EMA     = 0.97
PCB_EMA        = 0.95

MAX_GLOBAL     = 200
MAX_PCB        = 300


def train(model, train_loader, val_loader, optimizer, num_epochs, save_dir, patience=5):

    print("\n Starting Two-Layer Dictionary Training...\n")
    dict_logger = DictLogger(save_dir)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []


    for epoch in range(num_epochs):
        model.train()
        dict_logger.reset_epoch()
        running_train_loss = 0.0

        #Track dictionary usage
        global_merge = 0
        pcb_merge = 0
        global_insert = 0
        pcb_insert = 0
        warmup = (epoch < 3)


        print(f"\n[EPOCH {epoch}] warmup={warmup}")

        for imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = imgs.cuda()

            # forward call now returns new format:
            # (best_sim_g, best_sim_p, recon_vals, q, new_keys, new_vals)
            total_loss, new_k, new_v, sim_g, sim_p, best_idx_g, best_idx_p = model(imgs)

            # backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if warmup:
                running_train_loss += total_loss.item()
                continue

            #DICTIONARY UPDATE(GLOBAL + PCB)
            with torch.no_grad():

                if sim_g is None:
                    bootstrap_update(
                        model=model,
                        new_k=new_k,
                        new_v=new_v,
                        dict_logger=dict_logger,
                        BOOTSTRAP_MERGE_G=BOOTSTRAP_MERGE_G,
                        BOOTSTRAP_MERGE_P=BOOTSTRAP_MERGE_P,
                        GLOBAL_EMA=GLOBAL_EMA,
                        PCB_EMA=PCB_EMA,
                    )
                    continue

                g_m, g_i, p_m, p_i = normal_update(
                    model=model,
                    new_k=new_k,
                    new_v=new_v,
                    sim_g=sim_g,
                    sim_p=sim_p,
                    best_idx_g=best_idx_g,
                    best_idx_p=best_idx_p,
                    dict_logger=dict_logger,
                    GLOBAL_MERGE_T=GLOBAL_MERGE_T,
                    PCB_MERGE_T=PCB_MERGE_T,
                    GLOBAL_EMA=GLOBAL_EMA,
                    PCB_EMA=PCB_EMA,
                    MAX_GLOBAL=MAX_GLOBAL,
                    MAX_PCB=MAX_PCB,
                )

                global_merge += g_m
                global_insert += g_i
                pcb_merge += p_m
                pcb_insert += p_i


            running_train_loss += total_loss.item()

        #End training loop
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"\nTrain Loss: {avg_train_loss:.6f}")
        print(f"Global Merge: {global_merge} | Global Insert: {global_insert}")
        print(f"PCB Merge: {pcb_merge} | PCB Insert: {pcb_insert}")
        print(f"D Sizes → Global: {model.dictionary.global_keys.shape[0]}, PCB: {model.dictionary.pcb_keys.shape[0]}")


        #VALIDATION
        if not warmup:
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for imgs in val_loader:
                    imgs = imgs.cuda()
                    total_loss, *_ = model(imgs)
                    running_val_loss += total_loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Val Loss: {avg_val_loss:.6f}")

            #EARLY STOPPING
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0

                save_path = os.path.join(save_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)

                print(f"Saved BEST model!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break
        

        #logging dict updation
        dict_logger.save_epoch(
            epoch=epoch,
            global_size=model.dictionary.global_keys.shape[0],
            pcb_size=model.dictionary.pcb_keys.shape[0],
            train_loss=avg_train_loss,
            val_loss=avg_val_loss if not warmup else None
        )

    #plotting the loss(train/val)
    plot_loss(train_losses, val_losses, save_dir)