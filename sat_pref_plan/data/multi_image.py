from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from sat_pref_plan.data.image_pair import (
    BaseImagePairDataset,
    ImagePairPyramid2PixelDataset,
)


class BaseMultiImagePairDataset(Dataset):
    def __init__(
        self,
        data_folder: Path,
        cache_dir: Path,
        patch_size: int,
        stride: int,
        embedder: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        num_workers: int = 4,
        custom_text: Optional[str] = None,
    ) -> None:
        self.data_folder = data_folder
        self.cache_dir = cache_dir
        self.patch_size = patch_size
        self.stride = stride
        self.embedder = embedder
        self.device = device
        self.num_workers = num_workers
        self.custom_text = custom_text if custom_text is not None else ""

        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

        self._load_data()

    def _load_data(self) -> None:
        # get pairs of images, load each as a pair dataset, then combine
        unmasked_image_paths, masked_image_paths = self._verify_data_folder()
        unmasked_images = sorted(unmasked_image_paths.iterdir())
        masked_images = sorted(masked_image_paths.iterdir())
        if len(unmasked_images) != len(masked_images):
            raise ValueError(
                f"Unequal number of unmasked and masked images in {self.data_folder}"
            )
        if not all([u.stem == m.stem for u, m in zip(unmasked_images, masked_images)]):
            raise ValueError(
                f"Unmasked and masked images do not match in {self.data_folder}"
            )

        # TODO: load each pair dataset in parallel
        Xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for u, m in zip(unmasked_images, masked_images):
            d = self._make_single_dataset(u, m)
            if d.X is not None and d.y is not None:
                Xs.append(d.X)
                ys.append(d.y)

        self.X = torch.cat(Xs)
        self.y = torch.cat(ys)

    def _verify_data_folder(self) -> Tuple[Path, Path]:
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
        if not self.data_folder.is_dir():
            raise NotADirectoryError(
                f"Data folder is not a directory: {self.data_folder}"
            )

        unmasked_image_paths: Optional[Path] = None
        masked_image_paths: Optional[Path] = None
        for p in self.data_folder.iterdir():
            if p.is_dir():
                if p.stem == "unmasked":
                    unmasked_image_paths = p
                elif p.stem == "masked":
                    masked_image_paths = p

        if unmasked_image_paths is None or masked_image_paths is None:
            raise FileNotFoundError(
                f"Could not find unmasked and masked image subfolders in {self.data_folder}"
            )

        return unmasked_image_paths, masked_image_paths

    def _make_single_dataset(
        self,
        unmasked_image_path: Path,
        masked_image_path: Path,
    ) -> BaseImagePairDataset:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.X is None or self.y is None:
            raise RuntimeError("Tried to access data before loading it!")
        return self.X[index], self.y[index]


class MultiImagePairPyramid2PixelDataset(BaseMultiImagePairDataset):
    def _make_single_dataset(
        self, unmasked_image_path: Path, masked_image_path: Path
    ) -> BaseImagePairDataset:
        return ImagePairPyramid2PixelDataset(
            unmasked_image_path,
            masked_image_path,
            self.cache_dir,
            self.patch_size,
            self.stride,
            self.embedder,
            num_levels=3,
            device=self.device,
            num_workers=self.num_workers,
        )
