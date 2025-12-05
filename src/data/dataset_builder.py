from torch.utils.data import ConcatDataset
from .simple_dataset import SimpleImageDataset
from .dataset_paths import get_dataset_paths
from .augmentations import build_transforms


def build_dataset(cfg):

    """
        Returns a combined dataset (ConcatDataset),
        without creating DataLoaders.
    """

    dataset_names = cfg.data.dataset_names
    dataset_roots = cfg.data.dataset_roots

    transform = build_transforms(
        image_size=cfg.data.img_size,
        multi_scale=cfg.data.get("multi_scale", False),
        strong_aug=cfg.data.get("strong_aug", False)
    )

    datasets = []
    total_count = 0

    for name in dataset_names:
        print(name)
        if name not in dataset_roots:
            print(f"[WARN] No root path for dataset: {name}")
            continue

        root = dataset_roots[name]
        subfolders = get_dataset_paths(name, root)

        ds = SimpleImageDataset(subfolders, transform)
        datasets.append(ds)
        total_count += len(ds)

        print(f"Loaded {len(ds)} images from {name}")

    if len(datasets) == 0:
        raise RuntimeError("No datasets loaded. Check dataset paths in config.")

    combined = ConcatDataset(datasets)
    print(f"Total training images from all datasets: {total_count}")

    return combined
