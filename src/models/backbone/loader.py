import torch


def load_backbone(cfg):
    """
        backbone loader.
        Example config:
            backbone:
                repo: "facebookresearch/dinov2"
                name: "dinov2_vitb14"
                frozen: true
    """

    repo = cfg.backbone.repo      # e.g., "facebookresearch/dinov2"
    name = cfg.backbone.name      # e.g., "dinov2_vitb14"
    frozen = cfg.backbone.frozen
    device = cfg.device

    print(f"[Backbone] Loading {name} from repo {repo}")

    model = torch.hub.load(repo, name)

    #Freeze or unfreeze
    for p in model.parameters():
        p.requires_grad = not frozen

    if frozen:
        model.eval()

    model = model.to(device)

    print(f"[Backbone] Loaded: {name}  (Frozen={frozen})")

    return model
