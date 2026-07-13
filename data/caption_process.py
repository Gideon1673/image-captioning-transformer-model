import json
import re
from collections import Counter

from config import Config


# ==============================
# 1. Tokenizer đơn giản (word-level, tách theo khoảng trắng + dấu câu)
# ==============================
def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    # Tách riêng dấu câu ra khỏi từ, ví dụ "stairs." -> "stairs", "."
    text = re.sub(r"([.,!?])", r" \1 ", text)
    tokens = text.split()
    return tokens


# ==============================
# 2. Xây vocab từ TOÀN BỘ captions trong dataset
# ==============================
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def build_vocab(json_path: str, min_freq: int = 1):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()
    for filename, info in data.items():
        for caption in info["captions"]:
            tokens = simple_tokenize(caption)
            counter.update(tokens)

    # Chỉ giữ từ xuất hiện >= min_freq lần (loại từ hiếm/lỗi chính tả)
    words = [w for w, freq in counter.items() if freq >= min_freq]

    # Token đặc biệt luôn đứng đầu, ID cố định 0,1,2,3
    vocab = SPECIAL_TOKENS + sorted(words)

    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}

    return stoi, itos


# ==============================
# 3. Encode 1 caption -> list ID (có thêm <bos>, <eos>)
# ==============================
def encode_caption(caption: str, stoi: dict) -> list[int]:
    tokens = simple_tokenize(caption)
    ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    ids = [stoi["<bos>"]] + ids + [stoi["<eos>"]]
    return ids


# ==============================
# Kiểm tra nhanh
# ==============================
if __name__ == "__main__":
    stoi, itos = build_vocab(Config.image_to_captions_file)
    print("Vocab size:", len(stoi))

    sample_caption = "A child in a pink dress is climbing up a set of stairs in an entry way ."
    ids = encode_caption(sample_caption, stoi)

    print("Caption gốc:", sample_caption)
    print("Token IDs:", ids)
    print("Decode lại:", [itos[i] for i in ids])