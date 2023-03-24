from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import trange

from sat_pref_plan.data.utils import extract_patch_pair, extract_pyramid


class BaseImagePairDataset(Dataset):
    def __init__(
        self,
        unmasked_image_path: Path,
        masked_image_path: Path,
        cache_dir: Path,
        patch_size: int,
        stride: int,
        embedder: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        num_workers: int = 4,
        custom_text: Optional[str] = None,
    ) -> None:
        self.unmasked_image_path = unmasked_image_path
        self.masked_image_path = masked_image_path
        self.cache_dir = cache_dir
        self.stride = stride
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.embedder = embedder.to(device)
        self.device = device
        self.custom_text = custom_text if custom_text is not None else ""
        self.cache_path = (
            self.cache_dir
            / (self.__class__.__name__ + self.custom_text)
            / self.unmasked_image_path.stem
        )
        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self._load_data()

    def _load_data(self) -> None:
        # check cache for data, load it from there if possible
        cache_loaded = self._load_from_cache()

        # otherwise, load images and generate data and then save to cache
        if not cache_loaded:
            logger.info(
                f"Cached data not found, loading data from images: {self.unmasked_image_path}, {self.masked_image_path}"
            )
            self._load_from_images(
                read_image(self.unmasked_image_path.as_posix()),
                read_image(self.masked_image_path.as_posix()),
            )
            self._save_to_cache()

    def _load_from_cache(self) -> bool:
        if (
            self.cache_path.exists()
            and self.cache_path.is_dir()
            and len(list(self.cache_path.glob("*"))) > 0
        ):
            logger.info(f"Loading data from cache: {self.cache_path}")
            self.X = torch.load(self.cache_path / "X.pt")
            self.y = torch.load(self.cache_path / "y.pt")
            return True

        return False

    def _save_to_cache(self) -> None:
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)

        logger.info(f"Saving data to cache: {self.cache_path}")
        torch.save(self.X, self.cache_path / "X.pt")
        torch.save(self.y, self.cache_path / "y.pt")

    def _load_from_images(
        self, unmasked_image: torch.Tensor, masked_image: torch.Tensor
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.y) if self.y is not None else 0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.X is None or self.y is None:
            raise RuntimeError("Tried to access data before loading it!")
        return self.X[index], self.y[index]


# maps a (patch_size, patch_size) patch to a (patch_size, patch_size) patch
class ImagePairPatch2PatchDataset(BaseImagePairDataset):
    pass


# maps a (patch_size, patch_size) patch to a single representation value
class ImagePairPatch2PixelDataset(BaseImagePairDataset):
    pass


# maps a (num_layers, patch_size, patch_size) pyramid to a (patch_size, patch_size) patch
class ImagePairPyramid2PatchDataset(BaseImagePairDataset):
    pass


# maps a (num_layers, patch_size, patch_size) pyramid to a single representation value
class ImagePairPyramid2PixelDataset(BaseImagePairDataset):
    def __init__(
        self,
        unmasked_image_path: Path,
        masked_image_path: Path,
        cache_dir: Path,
        patch_size: int,
        stride: int,
        embedder: torch.nn.Module,
        num_levels: int = 3,
        device: torch.device = torch.device("cpu"),
        num_workers: int = 4,
    ) -> None:
        self.num_levels = num_levels
        super().__init__(
            unmasked_image_path,
            masked_image_path,
            cache_dir,
            patch_size,
            stride,
            embedder,
            device,
            num_workers,
            custom_text=str(num_levels),
        )

    def _load_from_images(
        self, unmasked_image: torch.Tensor, masked_image: torch.Tensor
    ) -> None:
        if unmasked_image.shape != masked_image.shape:
            logger.error(
                f"Unmasked image shape {unmasked_image.shape} does not match masked image shape {masked_image.shape}. Exiting."
            )
            exit(1)

        unmasked_image = unmasked_image.to(self.device)
        masked_image = masked_image.to(self.device)
        Xs = []
        ys = []
        with torch.no_grad():
            for i in trange(
                unmasked_image.shape[1]
                // self.stride
                * unmasked_image.shape[2]
                // self.stride,
                desc=f"Loading data from pair: {self.unmasked_image_path.name}",
            ):
                _, m = extract_patch_pair(
                    i, unmasked_image, masked_image, self.patch_size, self.stride
                )
                em: torch.Tensor = self.embedder(m)

                Xs.append(
                    extract_pyramid(
                        unmasked_image, i, self.patch_size, self.num_levels, self.stride
                    )
                )
                ys.append(em)

            self.X = torch.stack(Xs)
            self.y = torch.stack(ys)
