from typing import Tuple

import torch

from sat_pref_plan.data.base import PatchDataset, PyramidPatchDataset
from sat_pref_plan.data.utils import (
    compare_patches,
    extract_patch_pair,
    extract_pyramid,
)


class ModifiedPatchPatchDataset(PatchDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        l: int = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        og_patch, mod_patch = extract_patch_pair(
            index, self.image1, self.image2, self.patch_size, self.stride
        )
        return og_patch, compare_patches(og_patch, mod_patch).any()


class ModifiedPatchPixelDataset(PatchDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        l: int = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        og_patch, mod_patch = extract_patch_pair(
            index, self.image1, self.image2, self.patch_size, self.stride
        )
        return og_patch, compare_patches(og_patch, mod_patch)


class ModifiedPyramidPatchPatchDataset(PyramidPatchDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        l: int = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        og_patch, mod_patch = extract_patch_pair(
            index, self.image1, self.image2, self.patch_size, self.stride
        )

        return (
            extract_pyramid(
                self.image1, index, self.patch_size, self.num_levels, self.stride
            ),
            compare_patches(og_patch, mod_patch).any(),
        )


class ModifiedPyramidPatchPixelDataset(PyramidPatchDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        l: int = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        og_patch, mod_patch = extract_patch_pair(
            index, self.image1, self.image2, self.patch_size, self.stride
        )

        return (
            extract_pyramid(
                self.image1, index, self.patch_size, self.num_levels, self.stride
            ),
            compare_patches(og_patch, mod_patch),
        )
