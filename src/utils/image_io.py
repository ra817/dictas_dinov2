from PIL import Image
import os
import torchvision.transforms.functional as TF


def save_input_image(img_tensor, save_path):
    """
        Saves a single input tensor image into a .png file.
        Assumes tensor is shape [C, H, W] and in [0,1].
    """
    img = TF.to_pil_image(img_tensor.cpu())
    img.save(save_path)
