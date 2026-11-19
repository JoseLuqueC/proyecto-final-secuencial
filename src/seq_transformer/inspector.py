"""Instrumentación e inspección de tensores Q, K, V, atención y distribuciones FDR en D-FINE."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DFineForObjectDetection
from transformers.models.d_fine.modeling_d_fine import (
    DFineIntegral,
    DFineMultiscaleDeformableAttention,
    DFineSelfAttention,
)


@dataclass(frozen=True, slots=True)
class TensorStats:
    """Estadísticas resumidas de un tensor intermedio capturado."""

    shape: tuple[int, ...]
    dtype: str
    mean: float
    std: float
    min_val: float
    max_val: float
    l2_norm: float

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorStats:
        """Calcula estadísticas numéricas sin mantener referencias al grafo de autograd."""
        detached = tensor.detach().float()
        return cls(
            shape=tuple(int(dim) for dim in detached.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            mean=float(detached.mean().item()),
            std=float(detached.std(unbiased=False).item()),
            min_val=float(detached.min().item()),
            max_val=float(detached.max().item()),
            l2_norm=float(torch.linalg.vector_norm(detached).item()),
        )


@dataclass(frozen=True, slots=True)
class SelfAttentionQKV:
    """Tensores Q, K, V capturados en una capa `DFineSelfAttention`."""

    layer_name: str
    role: str
    hidden_input: TensorStats
    position_input: TensorStats | None
    query: TensorStats
    key: TensorStats
    value: TensorStats
    attn_output: TensorStats


@dataclass(frozen=True, slots=True)
class DeformableCrossCapture:
    """Captura de atención cruzada deformable en una capa del decoder."""

    layer_name: str
    query_input: TensorStats
    value_input: TensorStats
    sampling_offsets: TensorStats
    sampling_locations: np.ndarray
    attention_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class DetectionItem:
    """Detección individual asociada a su índice de Object Query en el decoder."""

    query_index: int
    label_id: int
    label_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(slots=True)
class InspectionResult:
    """Salida completa de inferencia e inspección interna sobre un frame."""

    image_size: tuple[int, int]
    latency_ms: float
    detections: list[DetectionItem] = field(default_factory=list)
    self_attn_qkv: list[SelfAttentionQKV] = field(default_factory=list)
    encoder_attentions: list[np.ndarray] = field(default_factory=list)
    decoder_self_attentions: list[np.ndarray] = field(default_factory=list)
    cross_deformable: list[DeformableCrossCapture] = field(default_factory=list)
    fdr_edge_probs: np.ndarray | None = None


def _compute_sampling_locations(
    module: DFineMultiscaleDeformableAttention,
    sampling_offsets: torch.Tensor,
    reference_points: torch.Tensor,
) -> torch.Tensor:
    """Replica el cálculo exacto de coordenadas normalizadas `(x, y)` de muestreo deformable."""
    if reference_points.shape[-1] == 4:
        num_points_scale = module.num_points_scale.to(
            dtype=sampling_offsets.dtype,
            device=sampling_offsets.device,
        ).unsqueeze(-1)
        offset = (
            sampling_offsets
            * num_points_scale
            * reference_points[:, :, None, :, 2:]
            * module.offset_scale
        )
        return reference_points[:, :, None, :, :2] + offset

    return reference_points[:, :, None, :, :2] + sampling_offsets


def _postprocess_with_query_indices(
    outputs: Any,
    target_size_hw: tuple[int, int],
    id2label: dict[int, str],
    threshold: float,
) -> list[DetectionItem]:
    """Convierte logits y cajas normalizadas `[cx, cy, w, h]` preservando el `query_index`."""
    logits = outputs.logits[0]  # (num_queries, num_labels)
    boxes = outputs.pred_boxes[0]  # (num_queries, 4) en [cx, cy, w, h] normalizado

    probs = torch.sigmoid(logits)
    scores, labels = probs.max(dim=-1)

    height, width = target_size_hw
    cx, cy, bw, bh = boxes.unbind(dim=-1)
    x1 = ((cx - 0.5 * bw) * width).clamp(0.0, float(width))
    y1 = ((cy - 0.5 * bh) * height).clamp(0.0, float(height))
    x2 = ((cx + 0.5 * bw) * width).clamp(0.0, float(width))
    y2 = ((cy + 0.5 * bh) * height).clamp(0.0, float(height))
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    keep_indices = torch.nonzero(scores >= threshold, as_tuple=False).squeeze(-1)
    sorted_order = torch.argsort(scores[keep_indices], descending=True)

    detections: list[DetectionItem] = []
    for idx in keep_indices[sorted_order].tolist():
        label_id = int(labels[idx].item())
        box_vals = boxes_xyxy[idx].detach().cpu().tolist()
        detections.append(
            DetectionItem(
                query_index=int(idx),
                label_id=label_id,
                label_name=id2label.get(label_id, f"class_{label_id}"),
                score=float(scores[idx].item()),
                box_xyxy=(
                    float(box_vals[0]),
                    float(box_vals[1]),
                    float(box_vals[2]),
                    float(box_vals[3]),
                ),
            )
        )
    return detections


class DFineInspector:
    """Ejecuta inferencia sobre `DFineForObjectDetection` capturando Q, K, V, atención y FDR."""

    def __init__(
        self,
        model: DFineForObjectDetection,
        processor: Any,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device or next(model.parameters()).device

    def inspect_image(
        self,
        image: Image.Image,
        *,
        threshold: float = 0.4,
        capture_internals: bool = True,
    ) -> InspectionResult:
        """Ejecuta una pasada forward sobre una imagen PIL y captura tensores internos."""
        rgb_image = image.convert("RGB")
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        qkv_records: list[SelfAttentionQKV] = []
        cross_records: list[DeformableCrossCapture] = []
        fdr_last_probs: list[np.ndarray] = []
        handles: list[Any] = []

        if capture_internals:
            handles = self._register_hooks(qkv_records, cross_records, fdr_last_probs)

        try:
            start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=capture_internals,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - start) * 1000.0
        finally:
            for handle in handles:
                handle.remove()

        width, height = rgb_image.size
        raw_id2label = self.model.config.id2label or {}
        detections = _postprocess_with_query_indices(
            outputs,
            target_size_hw=(height, width),
            id2label={int(k): str(v) for k, v in raw_id2label.items()},
            threshold=threshold,
        )

        enc_attns = (
            [att[0].detach().cpu().numpy() for att in outputs.encoder_attentions]
            if outputs.encoder_attentions
            else []
        )
        dec_attns = (
            [att[0].detach().cpu().numpy() for att in outputs.decoder_attentions]
            if outputs.decoder_attentions
            else []
        )

        return InspectionResult(
            image_size=(width, height),
            latency_ms=latency_ms,
            detections=detections,
            self_attn_qkv=qkv_records,
            encoder_attentions=enc_attns,
            decoder_self_attentions=dec_attns,
            cross_deformable=cross_records,
            fdr_edge_probs=fdr_last_probs[-1] if fdr_last_probs else None,
        )

    def _register_hooks(
        self,
        qkv_records: list[SelfAttentionQKV],
        cross_records: list[DeformableCrossCapture],
        fdr_last_probs: list[np.ndarray],
    ) -> list[Any]:
        """Registra forward hooks sin alterar parámetros ni flujo del modelo."""
        handles: list[Any] = []
        offsets_cache: dict[str, torch.Tensor] = {}

        for name, module in self.model.named_modules():
            if isinstance(module, DFineSelfAttention):
                role = "Encoder (AIFI)" if "encoder" in name else "Decoder (Self-Attn)"
                handles.append(
                    module.register_forward_hook(
                        self._make_self_attn_hook(name, role, qkv_records),
                        with_kwargs=True,
                    )
                )
            elif isinstance(module, DFineMultiscaleDeformableAttention):
                handles.append(
                    module.sampling_offsets.register_forward_hook(
                        self._make_offsets_hook(name, offsets_cache),
                    )
                )
                handles.append(
                    module.register_forward_hook(
                        self._make_cross_attn_hook(name, offsets_cache, cross_records),
                        with_kwargs=True,
                    )
                )
            elif isinstance(module, DFineIntegral):
                handles.append(
                    module.register_forward_hook(
                        self._make_fdr_hook(module.max_num_bins, fdr_last_probs),
                    )
                )

        return handles

    @staticmethod
    def _make_self_attn_hook(
        layer_name: str,
        role: str,
        records: list[SelfAttentionQKV],
    ) -> Any:
        def hook(
            module: DFineSelfAttention,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: tuple[torch.Tensor, ...],
        ) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            pos_embeds = kwargs.get("position_embeddings")
            if hidden_states is None:
                return

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            qk_input = hidden_states + pos_embeds if pos_embeds is not None else hidden_states

            q_tensor = module.q_proj(qk_input).view(hidden_shape).transpose(1, 2)
            k_tensor = module.k_proj(qk_input).view(hidden_shape).transpose(1, 2)
            v_tensor = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            records.append(
                SelfAttentionQKV(
                    layer_name=layer_name,
                    role=role,
                    hidden_input=TensorStats.from_tensor(hidden_states),
                    position_input=(
                        TensorStats.from_tensor(pos_embeds) if pos_embeds is not None else None
                    ),
                    query=TensorStats.from_tensor(q_tensor),
                    key=TensorStats.from_tensor(k_tensor),
                    value=TensorStats.from_tensor(v_tensor),
                    attn_output=TensorStats.from_tensor(output[0]),
                )
            )

        return hook

    @staticmethod
    def _make_offsets_hook(layer_name: str, cache: dict[str, torch.Tensor]) -> Any:
        def hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            cache[layer_name] = output.detach()

        return hook

    @staticmethod
    def _make_cross_attn_hook(
        layer_name: str,
        offsets_cache: dict[str, torch.Tensor],
        records: list[DeformableCrossCapture],
    ) -> Any:
        def hook(
            module: DFineMultiscaleDeformableAttention,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            encoder_states = kwargs.get("encoder_hidden_states")
            reference_points = kwargs.get("reference_points")
            raw_offsets = offsets_cache.get(layer_name)
            if (
                hidden_states is None
                or encoder_states is None
                or reference_points is None
                or raw_offsets is None
            ):
                return

            batch_size, num_queries, _ = hidden_states.shape
            offsets_reshaped = raw_offsets.reshape(
                batch_size,
                num_queries,
                module.n_heads,
                sum(module.num_points_list),
                2,
            )
            locations = _compute_sampling_locations(
                module,
                offsets_reshaped,
                reference_points,
            )
            attn_weights = output[1]

            records.append(
                DeformableCrossCapture(
                    layer_name=layer_name,
                    query_input=TensorStats.from_tensor(hidden_states),
                    value_input=TensorStats.from_tensor(encoder_states),
                    sampling_offsets=TensorStats.from_tensor(offsets_reshaped),
                    sampling_locations=locations[0].detach().cpu().numpy(),
                    attention_weights=attn_weights[0].detach().cpu().numpy(),
                )
            )

        return hook

    @staticmethod
    def _make_fdr_hook(max_num_bins: int, storage: list[np.ndarray]) -> Any:
        def hook(
            _module: DFineIntegral,
            args: tuple[Any, ...],
            _output: torch.Tensor,
        ) -> None:
            pred_corners = args[0]
            num_queries = pred_corners.shape[1]
            probs = F.softmax(
                pred_corners[0].reshape(num_queries, 4, max_num_bins + 1),
                dim=-1,
            )
            storage.append(probs.detach().cpu().numpy())

        return hook
