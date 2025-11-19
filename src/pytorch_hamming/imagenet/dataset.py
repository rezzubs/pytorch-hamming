import json
import logging
import typing
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast, override

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# NOTE: pytorch maintainers said they aren't interested in adding type stubs
# https://github.com/pytorch/vision/issues/2025#issuecomment-2296026610
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]

_logger = logging.getLogger(__name__)

type _Transform = Callable[[Image.Image], Tensor]

_default_transform = cast(
    _Transform,
    transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)


def _load_name_to_id(dataset_root: Path) -> dict[str, int]:
    """Loads the name-to-id mapping from the dataset root directory."""

    with open(dataset_root.joinpath("name_to_id.json")) as f:
        name_to_id = json.load(f)  # pyright: ignore[reportAny]
        if not isinstance(name_to_id, dict):
            raise ValueError("name_to_id.json is not a dictionary.")
        if not all(
            isinstance(key, str) and isinstance(value, int)
            for key, value in name_to_id.items()  # pyright: ignore[reportUnknownVariableType]
        ):
            raise ValueError("name_to_id.json contains invalid key-value pairs.")

        return cast(dict[str, int], name_to_id)


class ImageNet(Dataset[tuple[Tensor, int]]):
    """The ImageNet dataset.

    Args:
        data_path:
            Path to the dataset directory. Expects the following structure:
            - images: A directory containing jpg images
            - name_to_id.json: A dictionary containing a map from the image file
            name to the corresponding label ID.
        limit:
            Optional maximum number of images to load from the dataset. If None,
            all images are loaded.
        transform:
            Optional transformation function that overrides the default
            transform.
    """

    def __init__(
        self,
        data_path: Path,
        *,
        limit: int | None = None,
        transform: _Transform | None = None,
    ) -> None:
        _logger.info("Loading ImageNet from disk")

        if not data_path.exists():
            raise FileNotFoundError(f"Directory {data_path} does not exist.")

        if not data_path.is_dir():
            raise NotADirectoryError(f"{data_path} is not a directory.")

        if transform is None:
            transform = _default_transform

        name_to_id = _load_name_to_id(data_path)

        images_dir = data_path.joinpath("images")
        self._items: list[tuple[Tensor, int]] = []
        for index, entry in enumerate(images_dir.iterdir()):
            if limit is not None and index >= limit:
                _logger.debug(
                    f"Stopped loading ImageNet because the limit ({limit}) is reached."
                )
                break

            image = Image.open(entry).convert("RGB")
            try:
                label = name_to_id[entry.name]
            except KeyError:
                raise ValueError(f"Image {entry.name} not found in name_to_id.json.")
            self._items.append((transform(image), label))

        self._batches_cache: dict[
            tuple[int, torch.dtype, torch.device],
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = {}

    def __len__(self):
        return len(self._items)

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self._items[index]

    def batches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cache_key = (batch_size, dtype, device)

        if cache_key in self._batches_cache:
            _logger.debug(
                f"Using cached batch for batch_size={batch_size}, dtype={dtype}, device={device}"
            )
            return self._batches_cache[cache_key]

        _logger.info(
            f"Computing ImageNet batches for batch_size={batch_size}, \
dtype={dtype}, device={device}"
        )

        dataloader = typing.cast(
            DataLoader[list[torch.Tensor]],
            DataLoader(
                self,
                batch_size=batch_size,
                shuffle=False,
            ),
        )

        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        # NOTE: The following is necessary because pytorch doesn't provide a
        # type safe API for `DataLoader`.
        iterator = typing.cast(Iterator[list[torch.Tensor]], iter(dataloader))
        for batch in iterator:
            assert isinstance(batch, list)
            assert len(batch) == 2
            assert all(isinstance(x, torch.Tensor) for x in batch)

            batches.append(
                (batch[0].to(dtype).to(device), batch[1].to(dtype).to(device))
            )

        self._batches_cache[cache_key] = batches
        return batches
