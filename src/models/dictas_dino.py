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

        #Extract frozen DINOv2 features
        with torch.no_grad():
            feats_all = self.dinov2.get_intermediate_layers(
                imgs, n=self.layer_indices, reshape=False, return_class_token=False
            )
            feats_proc = [F.normalize(f, dim=-1) for f in feats_all]
            img_feats = torch.stack(feats_proc, dim=0).mean(0)

        B, N, D = img_feats.shape
        flat_feats = img_feats.reshape(B * N, D)


        #Project patches into dictionary space
        new_keys = F.normalize(self.dictionary.key_gen(flat_feats), dim=-1)
        new_vals = F.normalize(self.dictionary.val_gen(flat_feats), dim=-1)


        #Lookup from BOTH global & PCB dictionaries
        best_sim_g,  best_sim_p , recon_vals, q_proj, best_idx_g, best_idx_p = self.dictionary.lookup(flat_feats, 
                                                                                                      self.top_k, self.lookup)


        #If BOTH dictionaries empty: warmup
        if best_sim_g is None and best_sim_p is None:
            warmup_loss = (new_keys ** 2).mean()
            return warmup_loss, new_keys, new_vals, None, None, None, None

        #recon_vals = (B*N, D)
        retrieved = recon_vals.view(B, N, D)


        #Reconstruction loss
        diff = (q_proj - retrieved).pow(2).sum(-1)
        L_recon = diff.mean()


        #Smoothness loss
        h = int(N ** 0.5)
        diff_map = diff.view(B, h, h)

        L_smooth = (
            F.l1_loss(diff_map[:, :, 1:], diff_map[:, :, :-1]) +
            F.l1_loss(diff_map[:, 1:, :], diff_map[:, :-1, :])
        )


        #Alignment loss
        img_global = F.normalize(img_feats.mean(1), dim=-1)
        dict_global = F.normalize(self.dictionary.global_vals.mean(0, keepdim=True), dim=-1)

        L_align = 1 - (img_global * dict_global).sum(-1).mean()


        #Total loss
        total_loss = (L_recon + self.lambda_align * L_align + self.lambda_smooth * L_smooth)


        #Return all important values for training
        return total_loss, new_keys, new_vals, best_sim_g, best_sim_p, best_idx_g, best_idx_p
        
