from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.vocabulary import Vocabulary


IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.4633699059486389,
            0.44778499007225037,
            0.406446248292923
        ],
        std=[
            0.2705262303352356,
            0.264015793800354,
            0.277614027261734
        ]
    )
])


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        vocabulary: Vocabulary,
        max_caption_length: int = 40,
        transform=IMAGE_TRANSFORM,
        caption_idx: int | None = None
    ):
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {json_path}"
            )

        with json_path.open(
            mode="r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Dữ liệu JSON phải là object."
            )

        self.items = list(data.items())
        self.vocabulary = vocabulary
        self.max_caption_length = max_caption_length
        self.transform = transform
        self.caption_idx = caption_idx

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int
    ) -> dict[str, object]:
        filename, image_data = self.items[index]

        resize_image_path = image_data.get(
            "resize_image_path"
        )

        if not resize_image_path:
            raise ValueError(
                f"Ảnh {filename} không có "
                f"resize_image_path."
            )

        with Image.open(resize_image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        captions = image_data.get("captions")

        if not isinstance(captions, list) or not captions:
            raise ValueError(
                f"Ảnh {filename} không có caption."
            )

        if self.caption_idx is None:
            selected_index = torch.randint(
                low=0,
                high=len(captions),
                size=(1,)
            ).item()
        else:
            selected_index = (
                self.caption_idx % len(captions)
            )

        caption = captions[selected_index]

        caption_ids = self.vocabulary.encode(
            caption=caption,
            max_length=self.max_caption_length
        )

        return {
            "filename": filename,
            "image": image_tensor,
            "caption": caption,
            "caption_ids": torch.tensor(
                caption_ids,
                dtype=torch.long
            )
        }