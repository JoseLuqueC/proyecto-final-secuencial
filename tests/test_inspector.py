"""Pruebas unitarias para inspección Q/K/V, atención deformable y distribuciones FDR de D-FINE."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from transformers import DFineForObjectDetection

from seq_transformer.inspector import DFineInspector
from seq_transformer.models import summarize_architecture


def test_resumen_arquitectura_reporta_dimensiones_reales(
    tiny_dfine_model: DFineForObjectDetection,
) -> None:
    # Arrange / Act
    summary = summarize_architecture(tiny_dfine_model, model_id="tiny-test")

    # Assert
    assert summary.model_id == "tiny-test"
    assert summary.d_model == 32
    assert summary.encoder_layers == 1
    assert summary.decoder_layers == 2
    assert summary.num_queries == 10
    assert summary.max_num_bins == 8
    assert summary.total_params > 0


def test_inspector_captura_qkv_atencion_y_fdr_en_forward(
    tiny_dfine_model: DFineForObjectDetection,
    stub_processor: Any,
) -> None:
    # Arrange
    inspector = DFineInspector(model=tiny_dfine_model, processor=stub_processor)
    image = Image.new("RGB", (128, 96), color=(110, 140, 170))

    # Act
    result = inspector.inspect_image(image, threshold=0.0, capture_internals=True)

    # Assert
    assert result.image_size == (128, 96)
    assert result.latency_ms > 0.0
    assert len(result.detections) == 10
    assert len(result.self_attn_qkv) == 3  # 1 encoder AIFI + 2 decoder self-attn
    assert result.self_attn_qkv[0].role == "Encoder (AIFI)"
    assert result.self_attn_qkv[1].role == "Decoder (Self-Attn)"
    assert result.self_attn_qkv[0].query.shape == result.self_attn_qkv[0].key.shape
    assert len(result.encoder_attentions) == 1
    assert len(result.decoder_self_attentions) == 2
    assert len(result.cross_deformable) == 2
    assert result.cross_deformable[0].sampling_locations.shape == (10, 4, 8, 2)
    assert result.cross_deformable[0].attention_weights.shape == (10, 4, 8)
    assert result.fdr_edge_probs is not None
    assert result.fdr_edge_probs.shape == (10, 4, 9)
    assert np.allclose(result.fdr_edge_probs.sum(axis=-1), 1.0, atol=1e-4)
