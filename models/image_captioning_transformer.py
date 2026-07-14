from __future__ import annotations

import torch
import torch.nn as nn

from data.patching_embedding import PatchEmbedding
from data.positional_embedding import PositionalEmbedding
from models.language_modeling_head import LanguageModelingHead
from models.text_embedding import TextEmbedding
from models.text_transformer_decoder import TextTransformerDecoder
from models.vision_transformer_encoder import VisionTransformerEncoder


class ImageCaptioningTransformer(nn.Module):
    """
    Image Captioning Transformer hoàn chỉnh.

    Luồng xử lý:

    Image
    → Patch Embedding
    → Image Positional Embedding
    → Vision Transformer Encoder
    → Visual Features

    Caption IDs
    → Text Embedding
    → Text Transformer Decoder
    → Language Modeling Head
    → Logits
    """

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            vocabulary_size: int,
            max_caption_length: int,
            pad_token_id: int,
            d_model: int = 512,
            num_heads: int = 4,
            d_ff: int = 2048,
            num_encoder_layers: int = 4,
            num_decoder_layers: int = 4,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        patches_per_side = image_size // patch_size
        num_patches = patches_per_side ** 2

        self.patch_embedding = PatchEmbedding(patch_size=patch_size, in_channels=3, d_model=d_model)
        self.image_positional_embedding = PositionalEmbedding(num_patches=num_patches, d_model=d_model)
        self.vision_transformer_encoder = (
            VisionTransformerEncoder(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_layers=num_encoder_layers,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
        )

        self.text_embedding = TextEmbedding(
            vocabulary_size=vocabulary_size,
            d_model=d_model,
            max_sequence_length=max_caption_length,
            pad_token_id=pad_token_id,
            dropout=dropout
        )

        self.text_transformer_decoder = (
            TextTransformerDecoder(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_layers=num_decoder_layers,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
        )

        self.language_modeling_head = LanguageModelingHead(d_model=d_model, vocabulary_size=vocabulary_size)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, image_size, image_size]
        output: [B, num_patches, d_model]
        """

        patch_tokens = self.patch_embedding(images)
        image_tokens = self.image_positional_embedding(patch_tokens)
        visual_features = self.vision_transformer_encoder(image_tokens)

        return visual_features

    def decode_captions(
            self,
            decoder_input_ids: torch.Tensor,
            visual_features: torch.Tensor,
            decoder_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        decoder_input_ids: [B, text_length]
        visual_features: [B, num_patches, d_model]
        output logits: [B, text_length, vocabulary_size]
        """

        text_embeddings = self.text_embedding(decoder_input_ids)

        decoder_features = (
            self.text_transformer_decoder(
                text_embeddings=text_embeddings,
                visual_features=visual_features,
                padding_mask=decoder_padding_mask
            )
        )

        logits = self.language_modeling_head(decoder_features)

        return logits

    def forward(
            self,
            images: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            decoder_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        visual_features = self.encode_images(images)

        logits = self.decode_captions(
            decoder_input_ids=decoder_input_ids,
            visual_features=visual_features,
            decoder_padding_mask=decoder_padding_mask
        )

        return logits
