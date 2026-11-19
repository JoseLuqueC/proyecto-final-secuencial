"""Catálogo y carga de modelos preentrenados D-FINE (Transformer Encoder-Decoder)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoImageProcessor, DFineConfig, DFineForObjectDetection

DFINE_MODELS: dict[str, str] = {
    "D-FINE-Nano (COCO · ligero)": "ustc-community/dfine_n_coco",
    "D-FINE-Small (COCO · balanceado)": "ustc-community/dfine_s_coco",
    "D-FINE-Medium (COCO · mayor precisión)": "ustc-community/dfine_m_coco",
    "D-FINE-Large (COCO · alta capacidad)": "ustc-community/dfine_l_coco",
    "D-FINE-XLarge (COCO · máxima precisión)": "ustc-community/dfine_x_coco",
}


@dataclass(frozen=True, slots=True)
class DFineArchitectureSummary:
    """Resumen estructural del modelo D-FINE cargado."""

    model_id: str
    d_model: int
    encoder_layers: int
    decoder_layers: int
    encoder_heads: int
    decoder_heads: int
    num_queries: int
    num_feature_levels: int
    decoder_n_points: list[int]
    max_num_bins: int
    num_labels: int
    total_params: int
    trainable_params: int
    device: str


def resolve_device(preferred: str | None = None) -> torch.device:
    """Resuelve el dispositivo de ejecución disponible (`cuda` o `cpu`)."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize_architecture(
    model: DFineForObjectDetection,
    model_id: str = "custom",
) -> DFineArchitectureSummary:
    """Extrae los hiperparámetros clave de un `DFineForObjectDetection`."""
    cfg: DFineConfig = model.config
    n_points = cfg.decoder_n_points
    points_list = (
        list(n_points)
        if isinstance(n_points, (list, tuple))
        else [int(n_points)] * int(cfg.num_feature_levels)
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device_str = str(next(model.parameters()).device)

    return DFineArchitectureSummary(
        model_id=model_id,
        d_model=int(cfg.d_model),
        encoder_layers=int(cfg.encoder_layers),
        decoder_layers=int(cfg.decoder_layers),
        encoder_heads=int(cfg.encoder_attention_heads),
        decoder_heads=int(cfg.decoder_attention_heads),
        num_queries=int(cfg.num_queries),
        num_feature_levels=int(cfg.num_feature_levels),
        decoder_n_points=points_list,
        max_num_bins=int(cfg.max_num_bins),
        num_labels=int(cfg.num_labels or 0),
        total_params=total_params,
        trainable_params=trainable_params,
        device=device_str,
    )


def load_dfine_model(
    model_id: str = "ustc-community/dfine_n_coco",
    *,
    attn_implementation: str = "eager",
    device: str | None = None,
    local_files_only: bool = False,
) -> tuple[Any, DFineForObjectDetection, torch.device]:
    """Carga el procesador de imagen y pesos preentrenados de D-FINE en modo evaluación."""
    target_device = resolve_device(device)
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model = DFineForObjectDetection.from_pretrained(
        model_id,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    model.to(target_device)
    model.eval()
    return processor, model, target_device
