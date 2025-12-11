import torch
import torch.nn as nn
import torch.nn.functional as F
from src.engine.losses import alignment_loss, reconstruction_loss, smoothness_loss


class DictAS_DINO(nn.Module):

    """DictA-S using a frozen DINOv2 encoder and learnable dictionary."""
    def __init__(self, dinov2_model, dictionary, lambda_align=0.05, lambda_smooth=0.05, layer_indices=[5, 11, 17, 23], lookup_temp=0.2, top_k=5):
        super().__init__()
        self.dinov2 = dinov2_model
        self.dictionary = dictionary
        self.lambda_align = lambda_align
        self.lambda_smooth = lambda_smooth
        self.layer_indices = layer_indices
        self.lookup = lookup_temp
        self.top_k = top_k



    #main model inference function
    def forward(self, imgs):

        #Extract DINO features
        with torch.no_grad():
            feats_all = self.dinov2.get_intermediate_layers(
                imgs, n=self.layer_indices, reshape=False, return_class_token=False
            )
            feats_proc = [F.normalize(f, dim=-1) for f in feats_all]
            img_feats = torch.stack(feats_proc, dim=0).mean(0)    # (B, N, D)

        B, N, D = img_feats.shape
        flat_feats = img_feats.reshape(B * N, D)

        #projected keys & values
        new_keys = F.normalize(self.dictionary.key_gen(flat_feats), dim=-1)
        new_vals = F.normalize(self.dictionary.val_gen(flat_feats), dim=-1)

        #Dictonary lookup
        best_sim, best_idx, proj_q = self.dictionary.lookup(flat_feats)
        print(best_sim)

        #If dictionary is empty, no reconstruction
        if best_sim is None:
            #Return zero loss to let trainer insert all keys
            warmup_loss = new_keys.pow(2).mean()
            return warmup_loss, 0, 0, 0, new_keys, new_vals, best_sim, best_idx

        #Compute retrieved features from dictionary
        retrieved = self.dictionary.keys[best_idx]     
        retrieved = retrieved.view(B, N, D)

        #Reconstruction loss
        diff = (img_feats - retrieved).pow(2).sum(-1)
        L_recon = diff.mean()

        #Smoothness
        h = int(N ** 0.5)
        diff_map = diff.view(B, h, h)

        L_smooth = (
            F.l1_loss(diff_map[:, :, 1:], diff_map[:, :, :-1]) +
            F.l1_loss(diff_map[:, 1:, :], diff_map[:, :-1, :])
        )

        #Global alignment loss
        img_global = F.normalize(img_feats.mean(1), dim=-1)
        dict_global = F.normalize(self.dictionary.keys.mean(0, keepdim=True), dim=-1)

        L_align = 1 - (img_global * dict_global).sum(-1).mean()

        #Total loss
        total_loss = L_recon + self.lambda_align * L_align + self.lambda_smooth * L_smooth

        return total_loss, L_recon, L_align, L_smooth, new_keys, new_vals, best_sim, best_idx

