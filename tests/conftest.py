"""Fixtures compartidos de pytest para pruebas offline de D-FINE."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import DFineConfig, DFineForObjectDetection


class StubImageProcessor:
    """Procesador mínimo sin red para pruebas deterministas offline."""

    def __call__(self, images: Image.Image, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        arr = np.asarray(images.resize((64, 64)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return {"pixel_values": tensor}


def build_tiny_dfine() -> DFineForObjectDetection:
    """Construye un `DFineForObjectDetection` mínimo en CPU sin descargar pesos."""
    cfg = DFineConfig(
        backbone_config={
            "model_type": "hgnet_v2",
            "num_channels": 3,
            "embedding_size": 8,
            "depths": [1, 1, 1],
            "hidden_sizes": [16, 16, 32],
            "stem_channels": [3, 8, 8],
            "stage_in_channels": [8, 16, 16],
            "stage_mid_channels": [8, 8, 16],
            "stage_out_channels": [16, 16, 32],
            "stage_num_blocks": [1, 1, 1],
            "stage_downsample": [False, True, True],
            "stage_downsample_strides": [2, 2, 2],
            "stage_light_block": [False, False, False],
            "stage_kernel_size": [3, 3, 3],
            "stage_numb_of_layers": [1, 1, 1],
            "stage_names": ["stem", "stage1", "stage2", "stage3"],
            "out_features": ["stage2", "stage3"],
            "out_indices": [2, 3],
        },
        encoder_in_channels=[16, 32],
        decoder_in_channels=[32, 32],
        feat_strides=[8, 16],
        encode_proj_layers=[1],
        encoder_hidden_dim=32,
        d_model=32,
        num_queries=10,
        encoder_layers=1,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_feature_levels=2,
        decoder_n_points=[4, 4],
        max_num_bins=8,
        id2label={0: "class_0", 1: "class_1", 2: "class_2"},
        eval_idx=-1,
    )
    cfg._attn_implementation = "eager"
    return DFineForObjectDetection(cfg).eval()


@pytest.fixture
def tiny_dfine_model() -> DFineForObjectDetection:
    """Retorna un modelo D-FINE mínimo en modo evaluación."""
    return build_tiny_dfine()


@pytest.fixture
def stub_processor() -> StubImageProcessor:
    """Retorna un procesador de imágenes determinista offline."""
    return StubImageProcessor()
