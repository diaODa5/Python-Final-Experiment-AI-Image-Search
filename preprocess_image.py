import numpy as np
from PIL import Image


def normalize(image_np):
    image_np = image_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = (image_np - mean) / std
    return image_np.transpose(2, 0, 1)[None]


def center_crop(img_path, crop_size=224):
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    left, top = (w - crop_size) // 2, (h - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return normalize(np.array(image))


def resize_short_side(img_path, target_size=224):
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # 计算缩放比例，使短边等于 224
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 按照 DINOv2 要求，长宽必须是 14 (patch_size) 的倍数
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14

    image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return normalize(np.array(image))