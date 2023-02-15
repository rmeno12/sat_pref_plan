from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms as T
import torch

# datasets for using two corresponding images:
# - matching patches
# - matching patches using image pyramid
# - yes/no is patch modified
# - yes/no is patch modified using image pyramid

# modifications include:
# - occlusion correction (also turning roofs into grass)
# - passability classification


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
    # print(img_patch.shape)
    # print(y - patch_size // 2, y + patch_size // 2)
    # print(x - patch_size // 2, x + patch_size // 2)
    a = 0 if y - patch_size // 2 > 0 else patch_size // 2 - y
    b = 0 if x - patch_size // 2 > 0 else patch_size // 2 - x
    patch[
        :,
        a : a + img_patch.shape[1],
        b : b + img_patch.shape[2],
    ] = img_patch

    return patch / 255


def extract_pyramid(image, index, patch_size, levels, stride) -> torch.Tensor:
    patches = torch.zeros(3 * levels, patch_size, patch_size)
    x = index % (image.shape[2] // stride) * stride
    y = index // (image.shape[2] // stride) * stride

    for i in range(levels):
        p = patch_size * 2**i
        patches[3 * i : 3 * i + 3] = T.Resize(size=patch_size)(
            extract_patch(image, x, y, p)
        )
    return patches


def compare_patches(patch1, patch2):
    return (patch1 != patch2).any(axis=0)


class PatchDataset(Dataset):
    def __init__(
        self,
        image1_file: str,
        image2_file: str,
        patch_size: int,
        stride: int = None,
        transform=None,
        target_transform=None,
    ):
        self.image1 = read_image(image1_file)
        self.image2 = read_image(image2_file)
        assert (
            self.image1.shape == self.image2.shape
        ), "Images must have the same shape."

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return (self.image1.shape[1] // self.stride) * (
            self.image1.shape[2] // self.stride
        )


class PyramidPatchDataset(PatchDataset):
    def __init__(
        self,
        image1_file: str,
        image2_file: str,
        patch_size: int,
        num_levels: int = 3,
        stride: int = None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            image1_file, image2_file, patch_size, stride, transform, target_transform
        )
        self.num_levels = num_levels


class MatchingPatchDataset(PatchDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        l = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        x = index % (self.image1.shape[2] // self.stride)
        y = index // (self.image1.shape[2] // self.stride)

        return (
            extract_patch(
                self.image1,
                x * self.stride,
                y * self.stride,
                self.patch_size,
                self.stride,
            ),
            extract_patch(self.image2, x, y, self.patch_size, self.stride),
        )


class MatchingPyramidPatchDataset(PyramidPatchDataset):
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        l = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        x = index % (self.image1.shape[2] // self.stride)
        y = index // (self.image1.shape[2] // self.stride)

        return (
            extract_pyramid(
                self.image1, index, self.patch_size, self.num_levels, self.stride
            ),
            extract_patch(self.image2, x, y, self.patch_size, self.stride),
        )


class ModifiedPatchPatchDataset(PatchDataset):
    def __getitem__(self, index):
        l = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        x = index % (self.image1.shape[2] // self.stride) * self.stride
        y = index // (self.image1.shape[2] // self.stride) * self.stride

        og_patch = extract_patch(self.image1, x, y, self.patch_size)
        mod_patch = extract_patch(self.image2, x, y, self.patch_size)
        return og_patch, compare_patches(og_patch, mod_patch).any()


class ModifiedPatchPixelDataset(PatchDataset):
    def __getitem__(self, index) -> Tuple[torch.Tensor, bool]:
        l = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        x = index % (self.image1.shape[2] // self.stride) * self.stride
        y = index // (self.image1.shape[2] // self.stride) * self.stride

        og_patch = extract_patch(self.image1, x, y, self.patch_size)
        mod_patch = extract_patch(self.image2, x, y, self.patch_size)
        return og_patch, compare_patches(og_patch, mod_patch)


class ModifiedPyramidPatchPatchDataset(PyramidPatchDataset):
    def __getitem__(self, index) -> Tuple[torch.Tensor, bool]:
        l = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        x = index % (self.image1.shape[2] // self.stride) * self.stride
        y = index // (self.image1.shape[2] // self.stride) * self.stride

        og_patch = extract_patch(self.image1, x, y, self.patch_size)
        mod_patch = extract_patch(self.image2, x, y, self.patch_size)

        return (
            extract_pyramid(
                self.image1, index, self.patch_size, self.num_levels, self.stride
            ),
            compare_patches(og_patch, mod_patch).any(),
        )


class ModifiedPyramidPatchPixelDataset(PyramidPatchDataset):
    def __getitem__(self, index) -> Tuple[torch.Tensor, bool]:
        l = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        x = index % (self.image1.shape[2] // self.stride) * self.stride
        y = index // (self.image1.shape[2] // self.stride) * self.stride

        og_patch = extract_patch(self.image1, x, y, self.patch_size)
        mod_patch = extract_patch(self.image2, x, y, self.patch_size)

        return (
            extract_pyramid(
                self.image1, index, self.patch_size, self.num_levels, self.stride
            ),
            compare_patches(og_patch, mod_patch),
        )
