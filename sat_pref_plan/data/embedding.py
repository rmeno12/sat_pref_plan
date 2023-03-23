from pathlib import Path
from typing import Tuple, Union

import torch

from sat_pref_plan.data.base import PyramidPatchDataset
from sat_pref_plan.data.utils import extract_patch_pair, extract_pyramid
from sat_pref_plan.models.embedding_net import EmbeddingNet


class EmbeddingDataset(PyramidPatchDataset):
    def __init__(
        self,
        data_path: Path,
        patch_size: int,
        embedding_net: EmbeddingNet,
        device: torch.device,
        num_levels: int = 3,
        stride: Union[int, None] = None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            data_path, patch_size, num_levels, stride, transform, target_transform
        )
        self.device = device
        self.embedding_net = embedding_net


class PixelEmbeddingPyramidDataset(EmbeddingDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        l: int = len(self)
        if index >= l:
            raise IndexError(f"Index {index} out of range [0, {l}).")

        u, m = self._get_image_pair_from_index(index)
        _, mod_patch = extract_patch_pair(index, u, m, self.patch_size, self.stride)

        raise NotImplementedError("PixelEmbeddingPyramidDataset not implemented yet.")
        return (
            extract_pyramid(u, index, self.patch_size, self.num_levels, self.stride),
            self.embedding_net(mod_patch),
        )


class PatchEmbeddingPyramidDataset(EmbeddingDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        l: int = len(self)
        if index >= l:
            raise IndexError(f"Index {index} out of range [0, {l}).")

        u, m = self._get_image_pair_from_index(index)
        _, mod_patch = extract_patch_pair(index, u, m, self.patch_size, self.stride)

        with torch.no_grad():
            embed_patch = self.embedding_net(mod_patch[None, :].to(self.device))

        return (
            extract_pyramid(u, index, self.patch_size, self.num_levels, self.stride),
            embed_patch,
        )
