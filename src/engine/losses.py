import torch
import torch.nn.functional as F


def reconstruction_loss(patch_feats, retrieved):
    """
    L_recon = ||x - x_hat||^2
    """
    return ((patch_feats - retrieved).pow(2).sum(-1)).mean()



def smoothness_loss(diff, B):
    """
    Spatial smoothness constraint on patch error map.
    diff: [B, N]
    """
    N = diff.shape[1]
    h = int(N ** 0.5)

    if h * h != N:
        return torch.tensor(0.0, device=diff.device)

    diff_map = diff.view(B, h, h)

    L_smooth = (
        F.l1_loss(diff_map[:, :, 1:], diff_map[:, :, :-1]) +
        F.l1_loss(diff_map[:, 1:, :], diff_map[:, :-1, :])
    )
    return L_smooth



def alignment_loss(patch_feats, dict_keys):
    """
    Align global image feature mean to dictionary center.
    """
    img_global = F.normalize(patch_feats.mean(1), dim=-1)
    dict_mean = F.normalize(dict_keys.mean(0, keepdim=True), dim=-1)
    return 1 - (img_global * dict_mean).sum(-1).mean()



def total_dictas_loss(L_recon, L_smooth, L_align, lambda_smooth, lambda_align):
    """
    L_total = L_recon + λ1 * L_smooth + λ2 * L_align
    """
    return L_recon + lambda_smooth * L_smooth + lambda_align * L_align
