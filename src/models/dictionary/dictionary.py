import torch
import torch.nn as nn
import torch.nn.functional as F


class DictionaryModule(nn.Module):
    def __init__(self, feat_dim=768, key_dim=768, val_dim=768, dict_size=256):
        super().__init__()

        #projection layers to convert backbone embedding space(img) to dict embedding space
        self.key_gen = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),               #dropout is added to off some of the neurons(to avoid model from memorizing)
            nn.Linear(1024, key_dim)
        )

        self.val_gen = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, val_dim)
        )

        #Dictionary as buffer (NO GRADIENTS)
        self.register_buffer("keys", torch.empty(dict_size, key_dim))
        self.register_buffer("values", torch.empty(dict_size, val_dim))



    #LOOKUP
    def lookup(self, q_feats, merge_threshold=0.80):
        """
        q_feats: (N, D)
        returns:
            best_sim: (N,)
            best_idx: (N,) but only valid if dict not empty
            q: projected normalized keys (for update or insert)
        """

        #Projecting query feats
        q_proj = self.key_gen(q_feats)
        q = F.normalize(q_proj, dim=-1)

        #CASE 1: Dictionary empty
        if self.keys.numel() == 0:
            return None, None, q

        #Compute similarity to dictionary
        k = F.normalize(self.keys, dim=-1)
        sim = torch.matmul(q, k.T)            #N, dict_size)

        best_sim, best_idx = sim.max(dim=1)   #picked the top similiar keys(for respective patches) from the dict

        return best_sim, best_idx, q




    #EMA UPDATE of EXISTING ENTRY
    @torch.no_grad()
    def ema_update(self, idx, new_key, new_val, ema_decay=0.97):
        self.keys[idx] = ema_decay * self.keys[idx] + (1 - ema_decay) * new_key
        self.values[idx] = ema_decay * self.values[idx] + (1 - ema_decay) * new_val

        # Normalize to keep stable
        self.keys[idx] = F.normalize(self.keys[idx], dim=-1)
        self.values[idx] = F.normalize(self.values[idx], dim=-1)



    #INSERT NEW ENTRY (DYNAMIC GROW)
    @torch.no_grad()
    def insert(self, new_key, new_val):
        new_key = new_key.unsqueeze(0)
        new_val = new_val.unsqueeze(0)

        self.keys = torch.cat([self.keys, new_key], dim=0)
        self.values = torch.cat([self.values, new_val], dim=0)

        #print(f"[DynamicDict] Inserted new entry → new size = {self.keys.shape[0]}")
