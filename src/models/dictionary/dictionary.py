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
    def lookup(self, q_feats, top_k=5, temperature=0.2):
        """
        Returns:
            best_sim_g : (N,) max similarity to GLOBAL dict (None if empty)
            best_sim_p : (N,) max similarity to PCB dict (None if empty)
            recon_vals : (N, D) final reconstructed VALUES
            q          : (N, D) projected features
        """

        #PROJECT QUERY FEATURES
        q_proj = self.key_gen(q_feats)
        q = F.normalize(q_proj, dim=-1)
        N = q.shape[0]

        # CASE 0: BOTH DICTS ARE EMPTY
        if self.global_keys.shape[0] == 0 and self.pcb_keys.shape[0] == 0:
            return None, None, None, q, None, None

        #Normalize dictionary keys
        gk = F.normalize(self.global_keys, dim=-1) if self.global_keys.numel() > 0 else None
        pk = F.normalize(self.pcb_keys, dim=-1)    if self.pcb_keys.numel() > 0 else None

        #CASE 1: ONLY GLOBAL EXISTS
        if pk is None:
            sim_g = torch.matmul(q, gk.T)               
            topk_vals_g, topk_idx_g = sim_g.topk(k=min(top_k, sim_g.shape[1]), dim=1)

            weights_g = F.softmax(topk_vals_g / temperature, dim=1)
            recon_g = (weights_g.unsqueeze(-1) * self.global_vals[topk_idx_g]).sum(dim=1)

            best_sim_g = topk_vals_g[:, 0]
            best_idx_g = topk_idx_g[:, 0]
            return best_sim_g, None, recon_g, q_proj, best_idx_g, None


        # CASE 2: GLOBAL + PCB BOTH EXIST
        sim_g = torch.matmul(q, gk.T)             
        sim_p = torch.matmul(q, pk.T)             

        #Global top-k reconstruction
        topk_vals_g, topk_idx_g = sim_g.topk(k=min(top_k, sim_g.shape[1]), dim=1)
        weights_g = F.softmax(topk_vals_g / temperature, dim=1)
        recon_global = (weights_g.unsqueeze(-1) * self.global_vals[topk_idx_g]).sum(dim=1)
        best_sim_g = topk_vals_g[:, 0]
        best_idx_g = topk_idx_g[:, 0]

        #PCB top-k reconstruction
        topk_vals_p, topk_idx_p = sim_p.topk(k=min(top_k, sim_p.shape[1]), dim=1)
        weights_p = F.softmax(topk_vals_p / temperature, dim=1)
        recon_pcb = (weights_p.unsqueeze(-1) * self.pcb_vals[topk_idx_p]).sum(dim=1)
        best_sim_p = topk_vals_p[:, 0]
        best_idx_p = topk_idx_g[:, 0]


        #PICK PCB OR GLOBAL BASED ON WHICH IS MORE SIMILAR
        choose_pcb = best_sim_p > best_sim_g
        recon_vals = torch.where(
            choose_pcb.unsqueeze(-1),
            recon_pcb,
            recon_global
        )

        return best_sim_g,  best_sim_p , recon_vals, q, best_idx_g, best_idx_p




    #EMA UPDATE GLOBAL
    @torch.no_grad()
    def ema_update_global(self, idx, new_key, new_val, ema_decay=0.95):
        self.global_keys[idx] = (ema_decay * self.global_keys[idx] + (1 - ema_decay) * new_key)
        self.global_keys[idx] = F.normalize(self.global_keys[idx], dim=-1)

        ema_decay += 0.02
        self.global_vals[idx] = (ema_decay * self.global_vals[idx] + (1 - ema_decay) * new_val)
        self.global_vals[idx] = F.normalize(self.global_vals[idx], dim=-1)


    #EMA UPDATE PCB
    @torch.no_grad()
    def ema_update_pcb(self, idx, new_key, new_val, ema_decay=0.7):
        self.pcb_keys[idx] = (ema_decay * self.pcb_keys[idx] + (1 - ema_decay) * new_key)
        self.pcb_keys[idx] = F.normalize(self.pcb_keys[idx], dim=-1)

        ema_decay += 0.02
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

        print(f"[Global-INSERT] New Global dict size: {self.global_keys.shape[0]}")


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
