from __future__ import annotations

import re


class CaptionTokenizer:
    """
    Tokenizer đơn giản dành cho caption tiếng Anh.

    Ví dụ:

    "A girl sits in a pool ."

    trở thành:

    ["a", "girl", "sits", "in", "a", "pool"]
    """

    TOKEN_PATTERN = re.compile(
        r"[a-z0-9]+(?:'[a-z0-9]+)?"
    )

    def tokenize(self, caption: str) -> list[str]:
        if not isinstance(caption, str):
            raise TypeError(
                f"caption phải là str, nhận được "
                f"{type(caption).__name__}."
            )

        normalized_caption = caption.lower().strip()

        return self.TOKEN_PATTERN.findall(
            normalized_caption
        )
