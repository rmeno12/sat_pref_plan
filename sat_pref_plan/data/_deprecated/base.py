from pathlib import Path
from typing import Tuple, Union

import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.io import read_image


class PatchDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        patch_size: int,
        stride: Union[int, None] = None,
        transform=None,
        target_transform=None,
    ):
        unmasked_paths = sorted(data_path.glob("unmasked/*"))
        masked_paths = sorted(data_path.glob("masked/*"))
        if len(unmasked_paths) != len(masked_paths):
            logger.error(
                f"Number of unmasked ({len(unmasked_paths)}) and masked ({len(masked_paths)}) images don't match!"
            )
            exit(1)

        self.unmasked = [read_image(str(p)) for p in unmasked_paths]
        self.masked = [read_image(str(p)) for p in masked_paths]

        to_remove = []
        for i, (u, m) in enumerate(zip(self.unmasked, self.masked)):
            if u.shape != m.shape:
                logger.warning(
                    f"Skipping {str(unmasked_paths[i])} and {str(masked_paths[i])} due to shape mismatch!"
                )
                to_remove.append(i)
        for i in reversed(to_remove):
            del self.unmasked[i]
            del self.masked[i]

        logger.info(f"Loaded {len(self.unmasked)} image pairs")
        if len(self.unmasked) == 0:
            logger.error("No valid image pairs found!")
            exit(1)

        self.stride = stride if stride is not None else patch_size
        self.num_patches = [self._num_patches(u) for u in self.unmasked]
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return sum(self.num_patches)

    def _num_patches(self, image: torch.Tensor) -> int:
        return (image.shape[1] // self.stride) * (image.shape[2] // self.stride)

    def _get_image_pair_from_index(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num = 0
        i: int
        for i in self.num_patches:
            if index < i:
                break
            else:
                index -= i
                num += 1
        return self.unmasked[num], self.masked[num]


class PyramidPatchDataset(PatchDataset):
    def __init__(
        self,
        data_path: Path,
        patch_size: int,
        num_levels: int = 3,
        stride: Union[int, None] = None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(data_path, patch_size, stride, transform, target_transform)
        self.num_levels = num_levels
