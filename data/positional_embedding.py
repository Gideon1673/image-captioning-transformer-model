import torch
import torch.nn as nn

from data.patching_image import PatchEmbedding


class PositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding
    """
    def __init__(self, num_patches: int, d_model: int = 512):
        super().__init__()
        # nn.Embedding(num_patches, d_model): bảng tra cứu, mỗi vị trí patch
        # (0, 1, 2, ..., 195) có 1 vector d_model chiều RIÊNG, học được qua train
        self.position_embedding_table = nn.Embedding(num_patches, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, d_model) - output từ PatchEmbedding
        B, N, D = x.shape

        # Tạo chỉ số vị trí 0..N-1, đưa lên cùng device với x (quan trọng khi chạy GPU)
        positions = torch.arange(N, device=x.device)        # (num_patches,)

        pos_emb = self.position_embedding_table(positions)  # (num_patches, d_model)

        # Cộng vào embedding ảnh — broadcasting tự động nhân bản pos_emb
        # cho từng ảnh trong batch: (B, N, D) + (N, D) -> (B, N, D)
        return x + pos_emb


# ==============================
# Kiểm tra nhanh - Ráp toàn bộ pipeline ảnh
# ==============================
if __name__ == "__main__":
    dummy_batch = torch.randn(4, 3, 224, 224)  # batch 4 ảnh

    patch_size = 16
    d_model = 512
    num_patches = (224 // patch_size) ** 2  # = 196

    patch_embed = PatchEmbedding(patch_size=patch_size, in_channels=3, d_model=d_model)
    pos_embed = PositionalEmbedding(num_patches=num_patches, d_model=d_model)

    x = patch_embed(dummy_batch)   # (4, 196, 512) - chưa có thông tin vị trí
    x = pos_embed(x)               # (4, 196, 512) - đã cộng thêm vị trí

    print("Sau Patch Embedding:", patch_embed(dummy_batch).shape)  # torch.Size([4, 196, 512])
    print("Sau Positional Embedding:", x.shape)                     # torch.Size([4, 196, 512])
    print("Position table shape:", pos_embed.position_embedding_table.weight.shape)  # (196, 512)