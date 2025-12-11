import torch
import torch.nn as nn
import torch.nn.functional as F



class TwoLayerDictionaryModule(nn.Module):
    def __init__(self, feat_dim=768, key_dim=768, val_dim=768, global_init_size=0, pcb_init_size=0):
        super().__init__()

        #PROJECTION LAYERS(dinoV2 to dictionary feature space)
        self.key_gen = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, key_dim)
        )

        self.val_gen = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, val_dim)
        )


        #GLOBAL DICTIONARY(slow update)
        if global_init_size > 0:
            k = torch.randn(global_init_size, key_dim)
            v = torch.randn(global_init_size, val_dim)
            self.register_buffer("global_keys", F.normalize(k, dim=-1))
            self.register_buffer("global_vals", F.normalize(v, dim=-1))
        else:
            self.register_buffer("global_keys", torch.empty((0, key_dim)))
            self.register_buffer("global_vals", torch.empty((0, val_dim)))


        # PCB-SPECIFIC DICTIONARY(fast update)
        if pcb_init_size > 0:
            k = torch.randn(pcb_init_size, key_dim)
            v = torch.randn(pcb_init_size, val_dim)
            self.register_buffer("pcb_keys", F.normalize(k, dim=-1))
            self.register_buffer("pcb_vals", F.normalize(v, dim=-1))
        else:
            self.register_buffer("pcb_keys", torch.empty((0, key_dim)))
            self.register_buffer("pcb_vals", torch.empty((0, val_dim)))



    # LOOKUP(IN BOTH GLOBAL + PCB DICTIONARIES)
    def lookup(self, q_feats):
        """Return best match from BOTH layers."""

        #Project & normalize query
        q_proj = self.key_gen(q_feats)
        q = F.normalize(q_proj, dim=-1)

        #CASE 1: GLOBAL IS EMPTY + PCB IS EMPTY
        if self.global_keys.shape[0] == 0 and self.pcb_keys.shape[0] == 0:
            return None, None, None, q   #new dict must insert everything

        #ONLY GLOBAL EXISTS
        if self.pcb_keys.shape[0] == 0:
            sim_g = torch.matmul(q, F.normalize(self.global_keys, dim=-1).T)
            best_sim, best_idx = sim_g.max(dim=1)
            recon = self.global_keys[best_idx]
            return best_sim, best_idx, recon, q

        #CASE 3: BOTH EXIST
        gk = F.normalize(self.global_keys, dim=-1)
        pk = F.normalize(self.pcb_keys, dim=-1)

        sim_g = torch.matmul(q, gk.T)  # (N, G)
        sim_p = torch.matmul(q, pk.T)  # (N, P)

        best_sim_g, idx_g = sim_g.max(dim=1)
        best_sim_p, idx_p = sim_p.max(dim=1)

        #pick whichever similarity is bigger
        choose_pcb = best_sim_p > best_sim_g

        recon_global = self.global_keys[idx_g]
        recon_pcb = self.pcb_keys[idx_p]

        final_recon = torch.where(choose_pcb.unsqueeze(-1),recon_pcb, recon_global)

        return (best_sim_g, best_sim_p, final_recon, q)



    #EMA UPDATE GLOBAL
    @torch.no_grad()
    def update_global(self, idx, new_key, new_val, ema_decay):
        self.global_keys[idx] = (ema_decay * self.global_keys[idx] + (1 - ema_decay) * new_key)
        self.global_keys[idx] = F.normalize(self.global_keys[idx], dim=-1)

        self.global_vals[idx] = (ema_decay * self.global_vals[idx] + (1 - ema_decay) * new_val)
        self.global_vals[idx] = F.normalize(self.global_vals[idx], dim=-1)


    #EMA UPDATE PCB
    @torch.no_grad()
    def update_pcb(self, idx, new_key, new_val, ema_decay):
        self.pcb_keys[idx] = (ema_decay * self.pcb_keys[idx] + (1 - ema_decay) * new_key)
        self.pcb_keys[idx] = F.normalize(self.pcb_keys[idx], dim=-1)
        self.pcb_vals[idx] = (ema_decay * self.pcb_vals[idx] + (1 - ema_decay) * new_val)
        self.pcb_vals[idx] = F.normalize(self.pcb_vals[idx], dim=-1)



    #INSERT NEW PATCH INTO PCB DICTIONARY
    @torch.no_grad()
    def insert_global(self, new_key, new_val):
        """
        Insert a new (key, value) entry into the global dictionary.
        Normalizes both before insertion.
        """
        new_key = F.normalize(new_key, dim=-1).unsqueeze(0)
        new_val = F.normalize(new_val, dim=-1).unsqueeze(0)

        self.global_keys = torch.cat([self.global_keys, new_key], dim=0)
        self.global_vals = torch.cat([self.global_vals, new_val], dim=0)


    @torch.no_grad()
    def insert_pcb(self, new_key, new_val):
        """
        Insert a new (key, value) entry into the PCB-level dictionary.
        Normalizes both before insertion.
        """
        # Normalize new vectors
        new_key = F.normalize(new_key, dim=-1).unsqueeze(0)   
        new_val = F.normalize(new_val, dim=-1).unsqueeze(0)  

        # Append to dictionaries
        self.pcb_keys = torch.cat([self.pcb_keys, new_key], dim=0)
        self.pcb_vals = torch.cat([self.pcb_vals, new_val], dim=0)

        print(f"[PCB-INSERT] New PCB dict size: {self.pcb_keys.shape[0]}")
