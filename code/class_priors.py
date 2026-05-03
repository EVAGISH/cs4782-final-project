from pathlib import Path

import numpy as np
from PIL import Image


# Compressed stack of RGB uint8 images (N, H, W, 3). Extensible to video (N, T, H, W, 3) later.
CLASS_PRIORS_ARRAY_KEY = "images"


def pil_to_uint8_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_class_prior_images(path: str | Path) -> np.ndarray:
    npz_path = Path(path)
    if not npz_path.is_file():
        raise ValueError(f"Missing class prior archive: {npz_path}")

    with np.load(npz_path) as z:
        if CLASS_PRIORS_ARRAY_KEY not in z:
            raise ValueError(f"{npz_path} must contain an '{CLASS_PRIORS_ARRAY_KEY}' array")
        images = np.array(z[CLASS_PRIORS_ARRAY_KEY])

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected class priors with shape (N, H, W, 3), got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"Expected class priors to use dtype uint8, got {images.dtype}")
    return images


def append_class_prior_images(path: str | Path, new_images: np.ndarray) -> np.ndarray:
    npz_path = Path(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    if npz_path.is_file():
        existing = load_class_prior_images(npz_path)
        combined = np.concatenate([existing, new_images], axis=0)
    else:
        combined = new_images

    np.savez_compressed(npz_path, **{CLASS_PRIORS_ARRAY_KEY: combined})
    return combined


def count_class_prior_images(path: str | Path) -> int:
    npz_path = Path(path)
    if not npz_path.is_file():
        return 0
    return int(load_class_prior_images(npz_path).shape[0])
