import os
from torch.utils.data import Dataset
from PIL import Image


class SimpleImageDataset(Dataset):

    """
        Loads images from a list of directories (recursive).
        Applies transforms and returns only images.
    """
    
    def __init__(self, roots, transform=None, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        if isinstance(roots, str):
            roots = [roots]

        self.paths = []
        for root in roots:
            if not os.path.isdir(root):
                continue

            for dirpath, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(exts):
                        self.paths.append(os.path.join(dirpath, f))

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {roots}")

        self.transform = transform


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            raise RuntimeError(f"Unreadable file: {path}")

        if self.transform:
            img = self.transform(img)

        #return only image tensors
        return img
