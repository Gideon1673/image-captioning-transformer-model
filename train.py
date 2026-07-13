import torch
from torch.utils.data import DataLoader

from config import Config
from data.image_caption_dataset import ImageCaptionDataset
from data.patching_embedding import PatchEmbedding
from data.positional_embedding import PositionalEmbedding
# from config import Config
# from data.dataset import ImageTransform, load_transformed_images_to_gpu
from data.prepare_flicrk8k_datasets import prepare_dataset
from data.resize_image import resize_image
from data.split_dataset import split_dataset


# from models.module_vit_encoder import PatchEmbedding, ImageQKVProjection


def print_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    allocated_mb = (torch.cuda.memory_allocated() / 1024 ** 2)
    reserved_mb = (torch.cuda.memory_reserved() / 1024 ** 2)
    print(
        f"{label}: "
        f"allocated={allocated_mb:.2f} MB, "
        f"reserved={reserved_mb:.2f} MB"
    )


def main() -> None:
    # --------------------------------------------------
    # Bước 1: Chuẩn bị và chia dataset.
    # --------------------------------------------------
    # prepare_dataset()
    # split_dataset()
    # resize_image()

    # --------------------------------------------------
    # Bước 2: Chọn thiết bị huấn luyện.
    # --------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị đang sử dụng: {device}")

    # --------------------------------------------------
    # Bước 3: Khởi tạo Dataset.
    #
    # Trong ImageCaptionDataset:
    # - Đọc ảnh đã resize 224x224.
    # - ToTensor: [0,255] -> [0,1].
    # - Normalize bằng mean/std.
    # - Chọn ngẫu nhiên một caption.
    # --------------------------------------------------
    train_dataset = ImageCaptionDataset(json_path=Config.train_data_file)

    # --------------------------------------------------
    # Bước 4: Khởi tạo DataLoader.
    # --------------------------------------------------
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=False
    )

    print(f"Số ảnh huấn luyện: {len(train_dataset)}")
    print(f"Số batch: {len(train_dataloader)}")

    # --------------------------------------------------
    # Bước 5: Tính số lượng patch.
    #
    # Image: 224x224
    # Patch: 16x16
    #
    # Số patch mỗi chiều:
    # 224 / 16 = 14
    #
    # Tổng số patch:
    # 14 * 14 = 196
    # --------------------------------------------------
    patches_per_side = Config.image_size // Config.patch_size
    num_patches = patches_per_side ** 2

    # --------------------------------------------------
    # Bước 6: Khởi tạo Patch Embedding.
    #
    # Input:
    # [B, 3, 224, 224]
    #
    # Sau patchify:
    # [B, 196, 768]
    #
    # Sau Linear Projection:
    # [B, 196, d_model]
    # --------------------------------------------------
    patch_embedding = PatchEmbedding(
        patch_size=Config.patch_size,
        in_channels=3,
        d_model=Config.d_model
    ).to(device)

    print(f"Số patch trên mỗi ảnh: {num_patches}")

    # --------------------------------------------------
    # Bước 7: Khởi tạo Positional Embedding.
    #
    # Positional Embedding:
    # [196, d_model]
    #
    # Được cộng vào Patch Embedding:
    # [B, 196, d_model]
    # --------------------------------------------------
    positional_embedding = PositionalEmbedding(
        num_patches=num_patches,
        d_model=Config.d_model
    ).to(device)

    patch_embedding.train()
    positional_embedding.train()

    # --------------------------------------------------
    # Bước 8: Đưa từng batch qua Patch Embedding
    # và Positional Embedding.
    # --------------------------------------------------
    for batch_index, batch in enumerate(train_dataloader):
        images = batch["image"].to(device=device, non_blocking=True)

        captions = batch["caption"]
        filenames = batch["filename"]

        # images:
        # [B, 3, 224, 224]
        print(f"\nBatch {batch_index + 1}")
        print(f"Images: {images.shape}")

        # Chia patch và Linear Projection.
        patch_tokens = patch_embedding(images)

        # patch_tokens:
        # [B, 196, d_model]
        print(f"Patch embeddings: {patch_tokens.shape}")

        # Cộng positional embedding.
        image_tokens = positional_embedding(patch_tokens)

        # image_tokens:
        # [B, 196, d_model]
        print(f"Patch + positional embeddings: {image_tokens.shape}")

        # --------------------------------------------------
        # Kiểm tra shape.
        # --------------------------------------------------
        current_batch_size = images.shape[0]

        expected_shape = (current_batch_size, num_patches, Config.d_model)

        assert image_tokens.shape == expected_shape, (
            f"Shape không hợp lệ: "
            f"expected={expected_shape}, "
            f"actual={tuple(image_tokens.shape)}"
        )

        assert image_tokens.device == device, (
            f"Tensor nằm trên {image_tokens.device}, "
            f"nhưng model đang sử dụng {device}"
        )

        print(f"Caption đầu tiên: {captions[0]}")
        print(f"Filename đầu tiên: {filenames[0]}")
        print("Kiểm tra pipeline thành công.")

        # Tạm thời chỉ kiểm tra batch đầu tiên.
        # Xóa break khi thêm Vision Transformer Encoder
        # và vòng lặp huấn luyện hoàn chỉnh.
        break


if __name__ == "__main__":
    main()
