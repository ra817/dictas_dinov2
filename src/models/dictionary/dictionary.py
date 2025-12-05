import torch
import torch.nn as nn
import torch.nn.functional as F



class DictionaryModule(nn.Module):
    def __init__(self, feat_dim=1024, key_dim=1024, val_dim=1024, dict_size=2048):
        super().__init__()

        #two projection layers(keys & values) to push features map of encoder is similiar space of dictionary(keys & values)
        self.key_gen = nn.Sequential(
            nn.Linear(feat_dim, key_dim),
            nn.ReLU(),
            nn.Linear(key_dim, key_dim)
        )
        self.val_gen = nn.Sequential(
            nn.Linear(feat_dim, val_dim),
            nn.ReLU(),
            nn.Linear(val_dim, val_dim)
        )

        #randomanly intialize the key and values features 
        self.keys = nn.Parameter(torch.randn(dict_size, key_dim))
        self.values = nn.Parameter(torch.randn(dict_size, val_dim))
        nn.init.normal_(self.keys, std=0.02)
        nn.init.normal_(self.values, std=0.02)



    '''
    q_feats was intially shape: (1369,768)  x patch each of y dimension
    projected using the key projection layer
    normalized then and all the keys of the dict(4096,768)
    then do matrix multiplication of (1369,768) * (768,4096) to get 1369,4096)

    '''
    def lookup(self, q_feats, topk=5, temperature=0.15):
        """Retrieve dictionary vectors most similar to query features."""
        q_proj = self.key_gen(q_feats)
        q = F.normalize(q_proj, dim=-1)  
        k = F.normalize(self.keys, dim=-1)

        sim = torch.matmul(q, k.T)  #similarity (N, dict_size)
        topk_vals, topk_idx = sim.topk(k=topk, dim=-1)
        weights = F.softmax(topk_vals / temperature, dim=-1)

        v = self.values[topk_idx]  # retrieve values
        retrieved = (weights.unsqueeze(-1) * v).sum(dim=1)
        return retrieved, sim



