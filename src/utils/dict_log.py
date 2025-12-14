import os
import json
from collections import defaultdict


class DictLogger:
    def __init__(self, save_dir, filename="dict_update_stats.json"):
        self.path = os.path.join(save_dir, filename)

        # Initialize file if not exists
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({"epochs": []}, f, indent=4)

        # Per-epoch counters
        self.reset_epoch()

    def reset_epoch(self):
        """Reset counters at start of each epoch"""
        self.global_update_count = defaultdict(int)
        self.pcb_update_count = defaultdict(int)

    def log_global_update(self, idx):
        """Log one global dictionary key update"""
        self.global_update_count[int(idx)] += 1

    def log_pcb_update(self, idx):
        """Log one PCB dictionary key update"""
        self.pcb_update_count[int(idx)] += 1

    def save_epoch(
        self,
        epoch,
        global_size,
        pcb_size,
        train_loss=None,
        val_loss=None
    ):
        """Append epoch statistics to JSON file"""

        epoch_data = {
            "epoch": epoch,
            "dict_sizes": {
                "global": int(global_size),
                "pcb": int(pcb_size)
            },
            "global_update_count": dict(self.global_update_count),
            "pcb_update_count": dict(self.pcb_update_count),
        }

        if train_loss is not None:
            epoch_data["train_loss"] = float(train_loss)
        if val_loss is not None:
            epoch_data["val_loss"] = float(val_loss)

        # Load → append → save
        with open(self.path, "r") as f:
            all_data = json.load(f)

        all_data["epochs"].append(epoch_data)

        with open(self.path, "w") as f:
            json.dump(all_data, f, indent=4)
