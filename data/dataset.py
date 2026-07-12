from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from pathlib import Path
import json
from config import PROJECT_ROOT, TRAIN_DATA_FILE


class ImageTransform:
    """
    Tiền xử lý ảnh trước khi đưa vào Vision Transformer.
    Train mode: RGB -> resize cạnh ngắn -> random crop -> random horizontal flip -> Tensor -> normalize
    Validation/Test mode: RGB -> resize cạnh ngắn -> center crop -> Tensor -> normalize
    Output: Tensor có shape [3, image_size, image_size].
    """

    def __init__(
            self,
            image_size: int = 224,
            resize_size: int = 256,
            train: bool = True,
            horizontal_flip_probability: float = 0.5,
            mean: Sequence[float] = (0.5, 0.5, 0.5),
            std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size phải lớn hơn 0.")

        if resize_size < image_size:
            raise ValueError(
                "resize_size phải lớn hơn hoặc bằng image_size. "
                f"Nhận resize_size={resize_size}, "
                f"image_size={image_size}."
            )

        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability phải nằm trong [0, 1].")

        if len(mean) != 3:
            raise ValueError("mean phải chứa đúng 3 giá trị cho R, G, B.")

        if len(std) != 3:
            raise ValueError("std phải chứa đúng 3 giá trị cho R, G, B.")

        if any(value <= 0 for value in std):
            raise ValueError("Mọi giá trị trong std phải lớn hơn 0.")

        self.image_size = image_size
        self.resize_size = resize_size
        self.train = train

        self.horizontal_flip_probability = (
            horizontal_flip_probability
        )

        # Shape [3, 1, 1] để broadcast với Tensor [3, H, W].
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> Tensor:
        """
        Args:
            image: PIL Image với kích thước bất kỳ.

        Returns:
            Tensor float32 có shape: [3, image_size, image_size]
        """

        if not isinstance(image, Image.Image):
            raise TypeError(
                "ImageTransform yêu cầu đầu vào là PIL.Image.Image, "
                f"nhận được {type(image)}."
            )

        # Đảm bảo ảnh luôn có đúng ba channel R, G, B.
        image = image.convert("RGB")

        # Giữ tỉ lệ ảnh và resize cạnh ngắn về resize_size.
        image = self._resize_shorter_side(image)

        if self.train:
            image = self._random_crop(image)
            image = self._random_horizontal_flip(image)
        else:
            image = self._center_crop(image)

        # PIL Image [H, W, C] -> Tensor [C, H, W].
        image_tensor = self._to_tensor(image)

        # Normalize từng channel.
        image_tensor = self._normalize(image_tensor)

        return image_tensor

    def _resize_shorter_side(self, image: Image.Image) -> Image.Image:
        """
        Resize cạnh ngắn về resize_size, giữ nguyên tỉ lệ ảnh.
        Ví dụ: ảnh 500x375, resize_size=256 -> ảnh mới khoảng 341x256.
        """

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Kích thước ảnh không hợp lệ: width={width}, "
                f"height={height}."
            )

        if width < height:
            new_width = self.resize_size
            new_height = round(height * self.resize_size / width)
        else:
            new_height = self.resize_size
            new_width = round(width * self.resize_size / height)

        return image.resize(
            size=(new_width, new_height),
            resample=Image.Resampling.BILINEAR,
        )

    def _random_crop(self, image: Image.Image) -> Image.Image:
        """
        Lấy ngẫu nhiên một vùng image_size x image_size.
        Chỉ dùng cho tập train để tạo thêm biến thể ảnh.
        """

        width, height = image.size

        if width < self.image_size or height < self.image_size:
            raise ValueError(
                "Ảnh sau resize nhỏ hơn crop size. "
                f"Ảnh hiện tại: {width}x{height}, "
                f"crop size: {self.image_size}."
            )

        max_left = width - self.image_size
        max_top = height - self.image_size

        left = random.randint(0, max_left)
        top = random.randint(0, max_top)

        right = left + self.image_size
        bottom = top + self.image_size

        return image.crop((left, top, right, bottom))

    def _center_crop(self, image: Image.Image) -> Image.Image:
        """
        Lấy vùng chính giữa ảnh.
        Dùng cho validation và test để kết quả ổn định, không thay đổi giữa các lần đánh giá.
        """

        width, height = image.size

        if width < self.image_size or height < self.image_size:
            raise ValueError(
                "Ảnh sau resize nhỏ hơn crop size. "
                f"Ảnh hiện tại: {width}x{height}, "
                f"crop size: {self.image_size}."
            )

        left = (width - self.image_size) // 2
        top = (height - self.image_size) // 2

        right = left + self.image_size
        bottom = top + self.image_size

        return image.crop((left, top, right, bottom))

    def _random_horizontal_flip(self, image: Image.Image) -> Image.Image:
        """
        Lật ảnh theo chiều ngang với xác suất cấu hình.
        Không dùng vertical flip vì việc lật ngược ảnh theo chiều dọc thường tạo ra ảnh không tự nhiên.
        """

        should_flip = (random.random() < self.horizontal_flip_probability)

        if should_flip:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return image

    @staticmethod
    def _to_tensor(image: Image.Image, ) -> Tensor:
        """
        Chuyển PIL Image thành PyTorch Tensor.
        PIL/NumPy: [H, W, C], uint8, giá trị [0, 255]
        PyTorch: [C, H, W], float32, giá trị [0, 1]
        """

        image_array = np.asarray(image, dtype=np.float32)

        image_array = image_array.copy()

        image_tensor = torch.from_numpy(image_array)

        # [H, W, C] -> [C, H, W]
        image_tensor = image_tensor.permute(2, 0, 1)

        # [0, 255] -> [0, 1]
        image_tensor = image_tensor / 255.0

        return image_tensor.contiguous()

    def _normalize(self, image_tensor: Tensor) -> Tensor:
        """
        Normalize theo công thức: normalized = (pixel - mean) / std

        Với mean=0.5, std=0.5:
            0.0 -> -1.0
            0.5 ->  0.0
            1.0 ->  1.0
        """

        if image_tensor.ndim != 3:
            raise ValueError(
                "image_tensor phải có shape [C, H, W], "
                f"nhận được {tuple(image_tensor.shape)}."
            )

        if image_tensor.shape[0] != 3:
            raise ValueError(
                "image_tensor phải có đúng 3 channel RGB, "
                f"nhận được {image_tensor.shape[0]} channel."
            )

        return (image_tensor - self.mean) / self.std


def test_image_transform() -> None:
    with TRAIN_DATA_FILE.open(mode="r", encoding="utf-8") as file:
        train_mapping = json.load(file)

    transform = ImageTransform(
        image_size=224,
        resize_size=256,
        train=True,
    )

    image_tensors: list[torch.Tensor] = []

    # Tạm thời lấy 4 ảnh để kiểm tra.
    for image_name, image_record in list(train_mapping.items())[:4]:
        image_path = Path(image_record["image_path"])

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        with Image.open(image_path) as image:
            image_tensor = transform(image)

        print(
            f"{image_name}: "
            f"shape={tuple(image_tensor.shape)}, "
            f"dtype={image_tensor.dtype}, "
            f"min={image_tensor.min().item():.4f}, "
            f"max={image_tensor.max().item():.4f}"
        )

        image_tensors.append(image_tensor)

    images = torch.stack(image_tensors, dim=0)

    print()
    print("Batch shape:", images.shape)
    print("Batch dtype:", images.dtype)


def load_transformed_images_to_gpu(
        mapping_file: str | Path,
        transform: ImageTransform,
        batch_size: int = 16,
        device: torch.device | None = None,
) -> tuple[Tensor, list[str]]:
    """
    Đọc một batch ảnh từ file JSON, chạy ImageTransform và lưu batch kết quả vào GPU.
    Returns:
        images_gpu: Tensor ảnh nằm trên GPU, shape [B, 3, 224, 224].
        image_names: Danh sách tên ảnh tương ứng trong batch.
    """

    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")

    mapping_file = Path(mapping_file)

    if not mapping_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file mapping: {mapping_file}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mapping_file.open(mode="r", encoding="utf-8") as file:
        mapping: dict[str, dict] = json.load(file)

    image_tensors: list[Tensor] = []
    image_names: list[str] = []

    for image_name, image_record in mapping.items():
        image_path = Path(image_record["image_path"])

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.is_file():
            print(f"Bỏ qua ảnh không tồn tại: {image_path}")
            continue

        with Image.open(image_path) as image:
            # PIL Image -> Tensor [3, 224, 224] trên CPU.
            image_tensor = transform(image)

        image_tensors.append(image_tensor)
        image_names.append(image_name)

        if len(image_tensors) >= batch_size:
            break

    if not image_tensors:
        raise RuntimeError("Không tìm thấy ảnh hợp lệ để tạo batch.")

    # Danh sách B tensor [3, 224, 224] -> một tensor batch [B, 3, 224, 224].
    images_cpu = torch.stack(image_tensors, dim=0)

    # Chuyển toàn bộ batch từ RAM sang VRAM.
    images_gpu = images_cpu.to(device=device, non_blocking=True)

    return images_gpu, image_names
