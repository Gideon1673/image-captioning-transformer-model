from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


class ImageCaptionCollator:
    """
    Gom nhiều sample thành một batch và padding caption.

    PAD position trong caption_padding_mask có giá trị True.
    Token bình thường có giá trị False.
    """

    def __init__(
        self,
        pad_token_id: int
    ):
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        batch: list[dict[str, object]]
    ) -> dict[str, object]:
        images = torch.stack([
            item["image"]
            for item in batch
        ])

        caption_sequences = [
            item["caption_ids"]
            for item in batch
        ]

        caption_ids = pad_sequence(
            caption_sequences,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        caption_lengths = torch.tensor(
            [
                sequence.size(0)
                for sequence in caption_sequences
            ],
            dtype=torch.long
        )

        caption_padding_mask = (
            caption_ids == self.pad_token_id
        )

        return {
            "filenames": [
                item["filename"]
                for item in batch
            ],
            "images": images,
            "captions": [
                item["caption"]
                for item in batch
            ],
            "caption_ids": caption_ids,
            "caption_lengths": caption_lengths,
            "caption_padding_mask": (
                caption_padding_mask
            )
        }