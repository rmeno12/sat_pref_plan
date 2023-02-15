from typing import Tuple

import torch

from sat_pref_plan.data.base import PatchDataset, PyramidPatchDataset
from sat_pref_plan.data.utils import extract_patch_pair, extract_pyramid


class MatchingPatchDataset(PatchDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        l: int = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        return extract_patch_pair(
            index, self.image1, self.image2, self.patch_size, self.stride
        )


class MatchingPyramidPatchDataset(PyramidPatchDataset):
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        l: int = len(self)
        assert index < l, f"Index {index} out of range [0, {l})."

        _, mod_patch = extract_patch_pair(
            index, self.image1, self.image2, self.patch_size, self.stride
        )

        return (
            extract_pyramid(
                self.image1, index, self.patch_size, self.num_levels, self.stride
            ),
            mod_patch,
        )
