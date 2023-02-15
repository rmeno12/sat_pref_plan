from typing import Union
from torch.utils.data import Dataset
from torchvision.io import read_image


class PatchDataset(Dataset):
    def __init__(
        self,
        image1_file: str,
        image2_file: str,
        patch_size: int,
        stride: Union[int, None] = None,
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
        stride: Union[int, None] = None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            image1_file, image2_file, patch_size, stride, transform, target_transform
        )
        self.num_levels = num_levels
