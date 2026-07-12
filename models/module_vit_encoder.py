from __future__ import annotations

import torch
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """
    Chuyển batch ảnh thành chuỗi patch embeddings.
    Input: images: [B, 3, 224, 224]
    Output: image_tokens: [B, 196, 256]
    """

    def __init__(
            self,
            image_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 256,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size phải chia hết cho patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.patches_per_side = image_size // patch_size
        self.num_patches = self.patches_per_side ** 2

        self.patch_vector_dim = in_channels * patch_size * patch_size

        # [768] -> [256]
        self.projection = nn.Linear(self.patch_vector_dim, embed_dim)

        # Mỗi patch có một vector vị trí học được.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim, )
        )

        self.dropout = nn.Dropout(dropout)

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)

        nn.init.zeros_(self.projection.bias)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def patchify(self, images: Tensor) -> Tensor:
        """
        Input: [B, C, H, W]
        Output: [B, N, C*P*P]
        """

        if images.ndim != 4:
            raise ValueError("images phải có shape [B, C, H, W].")

        batch_size, channels, height, width = images.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Yêu cầu {self.in_channels} channels, "
                f"nhận được {channels}."
            )

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Yêu cầu ảnh {self.image_size}x"
                f"{self.image_size}, nhận được "
                f"{height}x{width}."
            )

        patch_size = self.patch_size

        # [B, C, H, W] -> [B, C, N_H, P, N_W, P]
        patches = images.reshape(
            batch_size,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )

        # [B, C, N_H, P, N_W, P] -> [B, N_H, N_W, C, P, P]
        patches = patches.permute(0, 2, 4, 1, 3, 5)

        # -> [B, 196, 768]
        patches = patches.contiguous().reshape(
            batch_size,
            self.num_patches,
            self.patch_vector_dim,
        )

        return patches

    def forward(self, images: Tensor) -> Tensor:
        # [B, 3, 224, 224] -> [B, 196, 768]
        patches = self.patchify(images)

        # [B, 196, 768] -> [B, 196, 256]
        patch_embeddings = self.projection(patches)

        # Thêm thông tin vị trí.
        image_tokens = patch_embeddings + self.position_embedding

        return self.dropout(image_tokens)


class ImageQKVProjection(nn.Module):
    """
    Tạo Query, Key và Value từ image tokens.
    Input: image_tokens: [B, N, D]

    Output:
        query: [B, H, N, Dh]
        key:   [B, H, N, Dh]
        value: [B, H, N, Dh]
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim phải chia hết cho num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.layer_norm = nn.LayerNorm(embed_dim)

        # Một lần tạo đồng thời Q, K và V.
        self.qkv_projection = nn.Linear(embed_dim, 3 * embed_dim)

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv_projection.weight)
        nn.init.zeros_(self.qkv_projection.bias)

    def forward(self, image_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if image_tokens.ndim != 3:
            raise ValueError("image_tokens phải có shape [B, N, D].")

        batch_size, num_patches, embed_dim = image_tokens.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Yêu cầu embed_dim={self.embed_dim}, "
                f"nhận được {embed_dim}."
            )

        # Pre-Norm trước Self-Attention.
        # [B, 196, 256]
        normalized_tokens = self.layer_norm(image_tokens)

        # [B, 196, 256] -> [B, 196, 768]
        qkv = self.qkv_projection(normalized_tokens)

        # [B, N, 3D] -> [B, N, 3, H, Dh]
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)

        # [B, N, 3, H, Dh] -> [3, B, H, N, Dh]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(dim=0)

        return query, key, value
