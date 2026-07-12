import torch

from config import TRAIN_DATA_FILE, Config
from data.dataset import ImageTransform, load_transformed_images_to_gpu
from data.prepare_flicrk8k_datasets import prepare_dataset
from data.split_dataset import split_dataset
from models.module_vit_encoder import PatchEmbedding, ImageQKVProjection


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
    prepare_dataset()
    split_dataset()

    # --------------------------------------------------
    # Bước 2: Khởi tạo ImageTransform.
    # --------------------------------------------------
    train_transform = ImageTransform(
        image_size=Config.image_size,
        resize_size=256,
        train=True,
        horizontal_flip_probability=0.5,
    )

    # --------------------------------------------------
    # Bước 3: Chọn GPU nếu CUDA khả dụng.
    # --------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    # --------------------------------------------------
    # Bước 4: Load và transform một batch ảnh.
    #
    # Output: images_gpu: [B, 3, 224, 224]
    # --------------------------------------------------
    images_gpu, image_names = load_transformed_images_to_gpu(
        mapping_file=TRAIN_DATA_FILE,
        transform=train_transform,
        batch_size=Config.batch_size,
        device=device
    )

    print("\n===== TRANSFORMED IMAGES =====")
    print("Image count:", len(image_names))
    print("Images shape:", images_gpu.shape)
    print("Images dtype:", images_gpu.dtype)
    print("Images device:", images_gpu.device)
    print("Images on CUDA:", images_gpu.is_cuda)

    assert images_gpu.ndim == 4
    assert images_gpu.shape[1:] == (3, 224, 224)

    print_cuda_memory("Sau khi load ảnh lên GPU")

    # --------------------------------------------------
    # Bước 5: Tạo PatchEmbedding trên GPU.
    # [B, 3, 224, 224] -> [B, 196, 256]
    # --------------------------------------------------
    patch_embedding = PatchEmbedding(
        image_size=Config.image_size,
        patch_size=Config.patch_size,
        in_channels=Config.in_channels,
        embed_dim=Config.embed_dim,
        dropout=Config.dropout,
    ).to(device)

    # --------------------------------------------------
    # Bước 6: Tạo module QKV Projection trên GPU.
    # [B, 196, 256] -> Q, K, V: [B, 4, 196, 64]
    # --------------------------------------------------
    qkv_projection = ImageQKVProjection(
        embed_dim=Config.embed_dim,
        num_heads=Config.encoder_heads,
    ).to(device)

    print("\n===== MODEL DEVICE =====")
    print("PatchEmbedding device:", next(patch_embedding.parameters()).device)

    print("QKV Projection device:", next(qkv_projection.parameters()).device)

    # Đây mới là kiểm tra forward, chưa huấn luyện.
    # eval() sẽ tắt Dropout.
    patch_embedding.eval()
    qkv_projection.eval()

    # --------------------------------------------------
    # Bước 7: Chuyển ảnh thành patch embeddings.
    # Bước 8: Tạo Q, K, V.
    # --------------------------------------------------
    with torch.no_grad():
        image_tokens = patch_embedding(images_gpu)
        query, key, value = qkv_projection(image_tokens)

        # --------------------------------------------------
        # Bước 9: Kiểm tra shapes.
        # --------------------------------------------------
        print("\n===== PATCH EMBEDDING =====")
        print("Image tokens shape:", image_tokens.shape)
        print("Image tokens device:", image_tokens.device)

        print("\n===== Q, K, V =====")
        print("Query shape:", query.shape)
        print("Key shape:", key.shape)
        print("Value shape:", value.shape)

        print("Query device:", query.device)
        print("Key device:", key.device)
        print("Value device:", value.device)

        batch_size = images_gpu.shape[0]

        expected_image_token_shape = (batch_size, 196, 256)

        expected_qkv_shape = (batch_size, 4, 196, 64)

        assert image_tokens.shape == expected_image_token_shape
        assert query.shape == expected_qkv_shape
        assert key.shape == expected_qkv_shape
        assert value.shape == expected_qkv_shape
        assert image_tokens.device == device
        assert query.device == device
        assert key.device == device
        assert value.device == device

        print_cuda_memory("Sau khi tạo Q, K, V")

        # --------------------------------------------------
        # Bước 10: Lấy Q, K, V của một ảnh cụ thể.
        # --------------------------------------------------
        first_image_name = image_names[0]

        # Shape mỗi ảnh:
        # [num_heads, num_patches, head_dim]
        # [4, 196, 64]
        first_image_query = query[0]
        first_image_key = key[0]
        first_image_value = value[0]

        print("\n===== FIRST IMAGE =====")
        print("Image name:", first_image_name)
        print("Query của ảnh đầu tiên:", first_image_query.shape)
        print("Key của ảnh đầu tiên:", first_image_key.shape)
        print("Value của ảnh đầu tiên:", first_image_value.shape)

        # Vector Query của:
        # - ảnh đầu tiên
        # - head đầu tiên
        # - patch đầu tiên
        #
        # Shape: [64]
        first_patch_query_vector = query[0, 0, 0,]
        print("Query vector của patch đầu tiên:", first_patch_query_vector.shape)
        print("\nĐã tạo Q, K, V thành công trên", device)


if __name__ == "__main__":
    main()
