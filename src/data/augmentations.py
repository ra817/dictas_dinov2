
import random
from torchvision import transforms


def build_transforms(image_size=224, multi_scale=False, strong_aug=False):
    
    """
        Build torchvision transforms for training.
        Supports multi-scale and adjustable augmentation strength.
        
        Args:
            image_size (int): Base image size (e.g., 224 or 336).
            multi_scale (bool): If True, randomly picks one from multiple sizes.
            strong_aug (bool): If True, applies stronger color/rotation jittering.
    """
    
    # Define multi-scale options (for defect scale robustness)
    size_list = [224, 256, 336, 384] if multi_scale else [image_size]
    chosen_size = random.choice(size_list)

    # Color / spatial augmentation strength
    cj_brightness = 0.2 if strong_aug else 0.1
    cj_contrast = 0.2 if strong_aug else 0.1
    cj_saturation = 0.2 if strong_aug else 0.1
    cj_hue = 0.04 if strong_aug else 0.02
    rot_deg = 10 if strong_aug else 5

    transform = transforms.Compose([
        transforms.Resize((chosen_size, chosen_size)),
        transforms.RandomResizedCrop(chosen_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=rot_deg),
        transforms.ColorJitter(
            brightness=cj_brightness,
            contrast=cj_contrast,
            saturation=cj_saturation,
            hue=cj_hue
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    return transform
