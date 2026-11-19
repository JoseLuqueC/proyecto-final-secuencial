"""Pruebas unitarias del pipeline de video y funciones de visualización."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import DFineForObjectDetection

from seq_transformer.inspector import (
    DeformableCrossCapture,
    DetectionItem,
    DFineInspector,
    TensorStats,
)
from seq_transformer.video import (
    analyze_video_sequence,
    create_synthetic_demo_video,
    extract_video_frames,
)
from seq_transformer.visualization import (
    draw_detections,
    plot_deformable_cross_attention,
    plot_encoder_attention_map,
    plot_fdr_distributions,
    plot_video_metrics,
)


def test_pipeline_video_sintetico_extrae_e_infiere_frames(
    tmp_path: Path,
    tiny_dfine_model: DFineForObjectDetection,
    stub_processor: Any,
) -> None:
    # Arrange
    video_path = create_synthetic_demo_video(tmp_path / "sample.mp4", num_frames=6)
    inspector = DFineInspector(model=tiny_dfine_model, processor=stub_processor)

    # Act
    frames, source_fps = extract_video_frames(video_path, stride=2, max_frames=5)
    analysis = analyze_video_sequence(
        video_path,
        inspector,
        threshold=0.0,
        stride=2,
        max_frames=3,
        capture_internals=False,
    )

    # Assert
    assert source_fps > 0.0
    assert len(frames) == 3
    assert len(analysis.frames_rgb) == 3
    assert len(analysis.metrics) == 3
    assert analysis.avg_latency_ms > 0.0
    assert analysis.avg_fps > 0.0


def test_visualizaciones_generan_figuras_validas(
    tmp_path: Path,
    tiny_dfine_model: DFineForObjectDetection,
    stub_processor: Any,
) -> None:
    # Arrange
    image = Image.new("RGB", (160, 120), color=(200, 210, 220))
    det = DetectionItem(
        query_index=2,
        label_id=0,
        label_name="car",
        score=0.87,
        box_xyxy=(20.0, 30.0, 100.0, 90.0),
    )
    enc_attn = np.full((4, 16, 16), 1.0 / 16.0, dtype=np.float32)
    dummy_stats = TensorStats((1, 10, 32), "float32", 0.0, 1.0, -1.0, 1.0, 5.0)
    cross_cap = DeformableCrossCapture(
        layer_name="layer.0",
        query_input=dummy_stats,
        value_input=dummy_stats,
        sampling_offsets=dummy_stats,
        sampling_locations=np.full((10, 4, 8, 2), 0.5, dtype=np.float32),
        attention_weights=np.full((10, 4, 8), 0.125, dtype=np.float32),
    )
    fdr_probs = np.full((10, 4, 9), 1.0 / 9.0, dtype=np.float32)

    video_path = create_synthetic_demo_video(tmp_path / "viz.mp4", num_frames=4)
    inspector = DFineInspector(model=tiny_dfine_model, processor=stub_processor)
    analysis = analyze_video_sequence(video_path, inspector, stride=1, max_frames=2)

    # Act
    annotated = draw_detections(image, [det], highlight_query_idx=2)
    fig_enc = plot_encoder_attention_map(image, enc_attn, token_row=1, token_col=2, head_idx=0)
    fig_cross = plot_deformable_cross_attention(
        image, cross_cap, query_idx=2, head_idx=0, box_xyxy=det.box_xyxy
    )
    fig_fdr = plot_fdr_distributions(fdr_probs, query_idx=2)
    fig_vid = plot_video_metrics(analysis.metrics)

    # Assert
    assert annotated.size == image.size
    assert fig_enc is not None
    assert fig_cross is not None
    assert fig_fdr is not None
    assert fig_vid is not None
    plt.close("all")
