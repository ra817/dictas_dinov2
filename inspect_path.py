import torch
import argparse

def inspect_pth(path):
    print(f"\n Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")

    print("\nTop-level keys in checkpoint:", ckpt.keys())

    # Print global_dict_size if present
    if "global_dict_size" in ckpt:
        print(f"\n🔍 global_dict_size = {ckpt['global_dict_size']}")
    else:
        print("\nNo global_dict_size found in checkpoint.")

    # Determine whether the checkpoint contains a state_dict
    if "state_dict" in ckpt:
        print("\n→ Using ckpt['state_dict'] for layer inspection.\n")
        state_dict = ckpt["state_dict"]
    else:
        print("\n→ Using entire checkpoint for inspection.\n")
        state_dict = ckpt

    print("Listing all parameters inside state_dict:\n")
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name:60s}  {tuple(tensor.shape)}")
        else:
            print(f"{name:60s}  (non-tensor)")

    print("\n Done. Total parameters listed:", len(state_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect .pth state dict")
    parser.add_argument("path", type=str, help="Path to .pth file")
    args = parser.parse_args()

    inspect_pth(args.path)


# import torch

# # Load checkpoints
# ckpt_large = torch.load("finetuned_models/Iter_1_checkpoint/best_model_V1.pth", map_location="cpu")
# ckpt_small = torch.load("finetuned_models/Iter_4_checkpoint_pcb_specific/best_model.pth", map_location="cpu")["state_dict"]

# # Extract keys & values
# keys_large = ckpt_large["dictionary.keys"]       # (4096, 768)
# vals_large = ckpt_large["dictionary.values"]     # (4096, 768)

# keys_small = ckpt_small["dictionary.keys"]       # (1024, 768)
# vals_small = ckpt_small["dictionary.values"]     # (1024, 768)

# # Concatenate
# keys_merged = torch.cat([keys_large, keys_small], dim=0)
# vals_merged = torch.cat([vals_large, vals_small], dim=0)

# print("Merged keys:", keys_merged.shape)     # (5120, 768)
# print("Merged values:", vals_merged.shape)   # (5120, 768)

# # Replace dictionary in checkpoint A
# ckpt_large["dictionary.keys"] = keys_merged
# ckpt_large["dictionary.values"] = vals_merged

# # Save merged checkpoint
# torch.save(ckpt_large, "checkpoint_merged.pth")
# print("Saved checkpoint_merged.pth")
