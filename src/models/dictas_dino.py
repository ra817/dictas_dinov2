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




    def forward(self, imgs):

        #Extract patch embeddings
        with torch.no_grad():
            feats_all = self.dinov2.get_intermediate_layers(
                imgs, n=self.layer_indices, reshape=False, return_class_token=False
            )
            feats_proc = [F.normalize(f, dim=-1) for f in feats_all]
            img_feats = torch.stack(feats_proc, dim=0).mean(0)   # (B, N, D)

        B, N, D = img_feats.shape
        flat_feats = img_feats.reshape(B * N, D)


        #Lookup normal dictionary reconstruction
        retrieved, sim = self.dictionary.lookup(flat_feats, topk=self.top_k, temperature=self.lookup)


        #Compute patch-level similarity score
        #top-k mean similarity = confidence of normal patch
        topk_vals, _ = sim.topk(k=self.top_k, dim=-1)
        patch_scores = topk_vals.mean(dim=-1)     

        threshold = patch_scores.mean() - 1.5 * patch_scores.std()
        threshold = threshold.clamp(min=0.2, max=0.5)
        print(threshold)

        #Create mask for NEW DICTIONARY ENTRY
        mask_new = patch_scores < 0.2

        #Replace reconstruction for new patches with trainable new values
        for idx in torch.where(mask_new)[0]:

            # Create new key/value pair 
            new_key = nn.Parameter(
                torch.randn(1, self.dictionary.keys.shape[-1], device=imgs.device) * 0.02
            )
            new_val = nn.Parameter(
                torch.randn(1, self.dictionary.values.shape[-1], device=imgs.device) * 0.02
            )

            # Add to dictionary
            self.dictionary.keys = nn.Parameter(
                torch.cat([self.dictionary.keys, new_key], dim=0)
            )
            self.dictionary.values = nn.Parameter(
                torch.cat([self.dictionary.values, new_val], dim=0)
            )

            retrieved[idx] = new_val.squeeze(0)

        #reshape reconstruction back to (B, N, D)
        retrieved = retrieved.view(B, N, D)


        # Compute losses
        diff = (img_feats - retrieved).pow(2).sum(-1)   # (B, N)
        L_recon = diff.mean()

        h = int(N**0.5)
        diff_map = diff.view(B, h, h)
        L_smooth = (
            F.l1_loss(diff_map[:, :, 1:], diff_map[:, :, :-1]) +
            F.l1_loss(diff_map[:, 1:, :], diff_map[:, :-1, :])
        )

        img_global = F.normalize(img_feats.mean(1), dim=-1)
        dict_mean = F.normalize(self.dictionary.keys.mean(0, keepdim=True), dim=-1)
        L_align = 1 - (img_global * dict_mean).sum(-1).mean()

        total_loss = L_recon + self.lambda_align * L_align + self.lambda_smooth * L_smooth

        return total_loss, L_recon, L_align, L_smooth