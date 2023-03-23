from typing import Tuple
import torch
import torchvision.transforms as T


# this breaks for odd patch sizes
# todo: fix this
def extract_patch(image: torch.Tensor, x: int, y: int, patch_size: int) -> torch.Tensor:
    # print(f"getting patch at ({x}, {y}) of size {patch_size})")
    patch = torch.zeros(3, patch_size, patch_size)
    img_patch = image[
        :,
        max(0, y - patch_size // 2) : min(y + patch_size // 2, image.shape[1]),
        max(0, x - patch_size // 2) : min(x + patch_size // 2, image.shape[2]),
    ]
    a = 0 if y - patch_size // 2 > 0 else patch_size // 2 - y
    b = 0 if x - patch_size // 2 > 0 else patch_size // 2 - x
    patch[
        :,
        a : a + img_patch.shape[1],
        b : b + img_patch.shape[2],
    ] = img_patch

    return patch / 255


def extract_pyramid(
    image: torch.Tensor, index: int, patch_size: int, levels: int, stride: int
) -> torch.Tensor:
    patches = torch.zeros(3 * levels, patch_size, patch_size)
    x = index % (image.shape[2] // stride) * stride
    y = index // (image.shape[2] // stride) * stride

    for i in range(levels):
        p = patch_size * 2**i
        patches[3 * i : 3 * i + 3] = T.Resize(size=patch_size)(
            extract_patch(image, x, y, p)
        )
    return patches


def compare_patches(patch1: torch.Tensor, patch2: torch.Tensor) -> torch.Tensor:
    return (patch1 != patch2).any(dim=0)


def extract_patch_pair(
    index: int, image1: torch.Tensor, image2: torch.Tensor, patch_size: int, stride: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = index % (image1.shape[2] // stride) * stride
    y = index // (image1.shape[2] // stride) * stride

    og_patch = extract_patch(image1, x, y, patch_size)
    mod_patch = extract_patch(image2, x, y, patch_size)

    return og_patch, mod_patch
