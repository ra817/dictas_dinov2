import os
import yaml
import torch
from torch.utils.data import random_split, DataLoader
from src.data.dataset_builder import build_dataset
from src.utils.config import load_config, set_seed
from src.engine.trainer import train
from src.models.backbone.loader import load_backbone
from src.models.dictas_dino import DictAS_DINO
from src.models.dictionary.dictionary import DictionaryModule



def main():

    #Load config
    cfg = load_config("config/config.yaml")
    set_seed(cfg.seed)

    #Device(CPU/GPU)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    #Output Directory
    os.makedirs(cfg.save_dir, exist_ok=True)


    #Data Loader
    full_dataset = build_dataset(cfg)

    #training/validation split
    val_ratio = cfg.training.val_split
    val_len = int(len(full_dataset) * val_ratio)
    train_len = len(full_dataset) - val_len

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    print(f"Total images: {len(full_dataset)}")
    print(f"Train: {train_len},  Val: {val_len}")

    #Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )



    #DINOv2 Backbone
    dinov2 = load_backbone(cfg)  #DinoV2-b
    num_params = sum(p.numel() for p in dinov2.parameters())
    print(f"DINOv2-B/14 parameters = {num_params:,}")



    #Model
    dictionary = DictionaryModule(
        feat_dim=cfg.Dictionary.feat_dim,
        key_dim=cfg.Dictionary.feat_dim,
        val_dim=cfg.Dictionary.feat_dim,
        dict_size=cfg.Dictionary.dict_size
    ).to(device)

    model = DictAS_DINO(
        dinov2_model=dinov2,
        dictionary=dictionary,
        lambda_align=cfg.Dictionary.lambda_align,
        lambda_smooth=cfg.Dictionary.lambda_smooth,
        layer_indices=cfg.backbone.get("layer_indices", None),
        lookup_temp = cfg.Dictionary.Lookup_Temp,
        top_k = cfg.Dictionary.Top_k

    ).to(device)


    #training continuity decision
    if cfg.training.continue_training:
        print(f"Continuing fine-tuning from {cfg.training.resume_checkpoint}")
        model.load_state_dict(torch.load(cfg.training.resume_checkpoint, map_location=device))
    else:
        print("Starting training from scratch...")


    optimizer = torch.optim.AdamW(model.dictionary.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)


    #Train
    train(model, train_loader, val_loader, optimizer, cfg.training.num_epochs, cfg.save_dir, cfg.training.patience)




if __name__ == "__main__":
    main()
