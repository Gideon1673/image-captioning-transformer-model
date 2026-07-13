from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import Config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RawImageDataset(Dataset):
    """
    Dataset chỉ dùng để TÍNH mean/std — chỉ áp dụng ToTensor(),
    """

    def __init__(self):
        self.image_dir = Path(Config.resize_image_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.image_dir}")

        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Đường dẫn không phải thư mục: {self.image_dir}")

        self.image_paths = [
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ]

        if not self.image_paths:
            raise RuntimeError(f"Không tìm thấy ảnh trong thư mục: {self.image_dir}")

        self.transform = transforms.ToTensor()

        print(
            f"Tìm thấy {len(self.image_paths)} ảnh "
            f"trong thư mục: {self.image_dir}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = self.transform(image)

            return image_tensor

        except Exception as exception:
            raise RuntimeError(f"Không thể đọc ảnh: {image_path}") from exception


def compute_mean_std(num_workers: int = 4):
    dataset = RawImageDataset()
    loader = DataLoader(
        dataset=dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    # Dùng float64 để giảm sai số khi cộng dồn hàng triệu pixel.
    channel_sum = torch.zeros(3, dtype=torch.float64)  # tổng giá trị pixel, từng kênh
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)  # tổng bình phương giá trị pixel, từng kênh
    total_pixels = 0

    for batch_index, images in enumerate(loader, start=1):
        # Shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = images.shape

        if channels != 3:
            raise ValueError(
                f"Ảnh phải có 3 channel RGB, "
                f"nhưng nhận được {channels} channel."
            )

        # Chuyển sang float64 để tính tổng chính xác hơn.
        images = images.to(dtype=torch.float64)

        # Cộng toàn bộ pixel theo các chiều: batch, height, width.
        # Kết quả còn lại có shape [3]: [tổng_R, tổng_G, tổng_B]
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += (images.square().sum(dim=(0, 2, 3)))
        total_pixels += batch_size * height * width

        print(f"\rĐang xử lý batch {batch_index}/{len(loader)}", end="", flush=True)

    print()

    if total_pixels == 0:
        raise RuntimeError("Không có pixel nào được xử lý.")

    mean = channel_sum / total_pixels

    # Var(X) = E[X²] - E[X]²
    variance = (channel_squared_sum / total_pixels - mean.square())

    # Tránh trường hợp xuất hiện số âm rất nhỏ do sai số số thực.
    variance = variance.clamp_min(0.0)
    std = torch.sqrt(variance)

    return mean.float(), std.float()


if __name__ == "__main__":
    mean, std = compute_mean_std(num_workers=41)

    print("Mean:", mean.tolist())
    print("Std:", std.tolist())

    print("\nSử dụng cho transforms.Normalize:")

    print(
        f"transforms.Normalize("
        f"mean={mean.tolist()}, "
        f"std={std.tolist()}"
        f")"
    )


# transforms.Normalize(mean=[0.4633699059486389, 0.44778499007225037, 0.406446248292923], std=[0.2705262303352356, 0.264015793800354, 0.277614027261734])