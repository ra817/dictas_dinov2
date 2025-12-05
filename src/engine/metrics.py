import torch
import torch.nn.functional as F


#Basic Metrics
def mse(x, y):
    return ((x - y) ** 2).mean().item()


def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=-1).mean().item()


#Anomaly Score (Patch-Level)
def anomaly_map(patch_feats, retrieved):
    """
    Compute per-patch anomaly score:
        A = ||x - x_hat||^2
    """
    diff = (patch_feats - retrieved).pow(2).sum(-1) # [B,N]
    return diff


#Global anomaly score
def global_anomaly_score(patch_feats, retrieved):
    """
    Global anomaly score = mean patch error.
    """
    diff = (patch_feats - retrieved).pow(2).sum(-1)
    return diff.mean(dim=-1)  # [B]
