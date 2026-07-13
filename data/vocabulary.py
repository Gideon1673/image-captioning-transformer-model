from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data.tokenizer import CaptionTokenizer


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    UNK_TOKEN_ID = 3

    def __init__(
        self,
        tokenizer: CaptionTokenizer
    ):
        self.tokenizer = tokenizer

        self.token_to_id: dict[str, int] = {
            self.PAD_TOKEN: self.PAD_TOKEN_ID,
            self.BOS_TOKEN: self.BOS_TOKEN_ID,
            self.EOS_TOKEN: self.EOS_TOKEN_ID,
            self.UNK_TOKEN: self.UNK_TOKEN_ID
        }

        self.id_to_token: dict[int, str] = {
            token_id: token
            for token, token_id
            in self.token_to_id.items()
        }

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.PAD_TOKEN_ID

    @property
    def bos_token_id(self) -> int:
        return self.BOS_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_TOKEN_ID

    @property
    def unk_token_id(self) -> int:
        return self.UNK_TOKEN_ID

    def build_from_json(
        self,
        json_path: str | Path,
        min_frequency: int = 2
    ) -> None:
        """
        Xây dựng vocabulary từ tất cả caption trong tập train.

        Chỉ sử dụng train dataset để tránh data leakage.
        """

        if min_frequency <= 0:
            raise ValueError(
                "min_frequency phải lớn hơn 0."
            )

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
                "Dữ liệu JSON phải là một object."
            )

        word_counter: Counter[str] = Counter()

        for image_name, image_data in data.items():
            if not isinstance(image_data, dict):
                raise ValueError(
                    f"Dữ liệu của {image_name} "
                    f"không phải object."
                )

            captions = image_data.get("captions")

            if not isinstance(captions, list):
                raise ValueError(
                    f"captions của {image_name} "
                    f"không phải list."
                )

            for caption in captions:
                tokens = self.tokenizer.tokenize(
                    caption
                )

                word_counter.update(tokens)

        # Sắp xếp để Vocabulary luôn được tạo giống nhau
        # giữa các lần chạy.
        valid_tokens = [
            token
            for token, frequency
            in word_counter.items()
            if frequency >= min_frequency
        ]

        valid_tokens.sort(
            key=lambda token: (
                -word_counter[token],
                token
            )
        )

        for token in valid_tokens:
            if token in self.token_to_id:
                continue

            token_id = len(self.token_to_id)

            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def encode(
        self,
        caption: str,
        max_length: int | None = None
    ) -> list[int]:
        """
        Chuyển caption thành token ID.

        Ví dụ:

        "a girl sits"

        trở thành:

        [BOS, a, girl, sits, EOS]
        """

        tokens = self.tokenizer.tokenize(caption)

        if max_length is not None:
            if max_length < 2:
                raise ValueError(
                    "max_length phải ít nhất bằng 2 "
                    "để chứa BOS và EOS."
                )

            # Chừa hai vị trí cho BOS và EOS.
            tokens = tokens[:max_length - 2]

        token_ids = [
            self.token_to_id.get(
                token,
                self.UNK_TOKEN_ID
            )
            for token in tokens
        ]

        return [
            self.BOS_TOKEN_ID,
            *token_ids,
            self.EOS_TOKEN_ID
        ]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Chuyển token ID về chuỗi văn bản.
        """

        tokens: list[str] = []

        special_tokens = {
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN
        }

        for token_id in token_ids:
            token = self.id_to_token.get(
                int(token_id),
                self.UNK_TOKEN
            )

            if token == self.EOS_TOKEN:
                break

            if (
                skip_special_tokens
                and token in special_tokens
            ):
                continue

            tokens.append(token)

        return " ".join(tokens)

    def save(
        self,
        output_path: str | Path
    ) -> None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        vocabulary_data = {
            "token_to_id": self.token_to_id
        }

        with output_path.open(
            mode="w",
            encoding="utf-8"
        ) as file:
            json.dump(
                vocabulary_data,
                file,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(
        cls,
        vocabulary_path: str | Path,
        tokenizer: CaptionTokenizer
    ) -> "Vocabulary":
        vocabulary_path = Path(vocabulary_path)

        if not vocabulary_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy vocabulary: "
                f"{vocabulary_path}"
            )

        with vocabulary_path.open(
            mode="r",
            encoding="utf-8"
        ) as file:
            vocabulary_data = json.load(file)

        vocabulary = cls(tokenizer=tokenizer)

        vocabulary.token_to_id = {
            str(token): int(token_id)
            for token, token_id
            in vocabulary_data["token_to_id"].items()
        }

        vocabulary.id_to_token = {
            token_id: token
            for token, token_id
            in vocabulary.token_to_id.items()
        }

        return vocabulary