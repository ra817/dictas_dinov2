import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict

# thresholds
BOOTSTRAP_MERGE_G = 0.95
BOOTSTRAP_MERGE_P = 0.96
GLOBAL_MERGE_T = 0.30      # global dictionary is more generic → low threshold
PCB_MERGE_T    = 0.70      # specific PCB → higher merge threshold
GLOBAL_EMA     = 0.99
PCB_EMA        = 0.90

MAX_GLOBAL     = 200
MAX_PCB        = 300


def train(model, train_loader, val_loader, optimizer, num_epochs, save_dir, patience=5):
    print("\n Starting Two-Layer Dictionary Training...\n")
    best_val_loss = float("inf")
    epochs_no_improve = 0


    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        #track dictionary usage
        global_merge = 0
        pcb_merge = 0
        global_insert = 0
        pcb_insert = 0
        warmup = (epoch<3)


        print(f"\n[EPOCH {epoch}] warmup={warmup}")

        for imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = imgs.cuda()

            # forward call now returns new format:
            # (best_sim_g, best_sim_p, recon_vals, q, new_keys, new_vals)
            total_loss, new_k, new_v, sim_g, sim_p, best_idx_g, best_idx_p = model(imgs)

            #backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if warmup:
                running_train_loss += total_loss.item()
                continue


            #DICTIONARY UPDATE(GLOBAL + PCB)
            with torch.no_grad():

                Bn = new_k.shape[0]

                #CASE_1: BOTH DICTS EMPTY: INSERT EVERYTHING AS GLOBAL
                if sim_g is None:
                    for i in range(Bn):
                        k_i = new_k[i]
                        v_i = new_v[i]

                        if model.dictionary.global_keys.shape[0] == 0:
                            model.dictionary.insert_global(k_i, v_i)
                            continue

                        if model.dictionary.pcb_keys.shape[0] == 0:
                            model.dictionary.insert_pcb(k_i, v_i)
                            continue

                        sims_g = torch.matmul(F.normalize(model.dictionary.global_keys, dim=-1),k_i.unsqueeze(-1)).squeeze(-1)
                        sims_p = torch.matmul(F.normalize(model.dictionary.pcb_keys, dim=-1),k_i.unsqueeze(-1)).squeeze(-1)


                        max_sim_g, idx_g = sims_g.max(dim=0)
                        max_sim_p, idx_p = sims_p.max(dim=0)


                        if max_sim_g >= BOOTSTRAP_MERGE_G:
                            model.dictionary.ema_update_global(idx_g, k_i, v_i, GLOBAL_EMA)
                        else:
                            model.dictionary.insert_global(k_i, v_i)

                        if max_sim_p >= BOOTSTRAP_MERGE_P:
                            model.dictionary.ema_update_pcb(idx_p, k_i, v_i, PCB_EMA)
                        else:
                            model.dictionary.insert_pcb(k_i, v_i)

                    continue


                for i in range(Bn):

                    k_i = new_k[i]
                    v_i = new_v[i]

                    #GLOBAL UPDATE
                    #If one patch of img got matched with dictionary keys(above threshold)
                    if sim_g[i] >= GLOBAL_MERGE_T:
                        idx_g = best_idx_g[i]
                        model.dictionary.ema_update_global(idx_g, k_i, v_i, GLOBAL_EMA)
                        global_merge += 1
                    else:
                        if model.dictionary.global_keys.shape[0] < MAX_GLOBAL:
                            model.dictionary.insert_global(k_i, v_i)
                            global_insert += 1

                    #PCB UPDATE
                    pcb_size = model.dictionary.pcb_keys.shape[0]
                    if sim_p[i] >= PCB_MERGE_T and best_idx_p[i] <  pcb_size:
                        idx_p = best_idx_p[i]
                        model.dictionary.ema_update_pcb(idx_p, k_i, v_i, PCB_EMA)
                        pcb_merge += 1
                    else:
                        if model.dictionary.pcb_keys.shape[0] < MAX_PCB:
                            model.dictionary.insert_pcb(k_i, v_i)
                            pcb_insert += 1

            running_train_loss += total_loss.item()

        #end training loop

        avg_train_loss = running_train_loss / len(train_loader)
        print(f"\nTrain Loss: {avg_train_loss:.6f}")
        print(f"Global Merge: {global_merge} | Global Insert: {global_insert}")
        print(f"PCB Merge: {pcb_merge} | PCB Insert: {pcb_insert}")
        print(f"D Sizes → Global: {model.dictionary.global_keys.shape[0]}, PCB: {model.dictionary.pcb_keys.shape[0]}")

        #VALIDATION
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.cuda()
                total_loss, *_ = model(imgs)
                running_val_loss += total_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
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


    #Save final model
    save_path = os.path.join(save_dir, "last_model.pth")
    torch.save(model.state_dict(), save_path)
    print("\nTraining Finished.")
