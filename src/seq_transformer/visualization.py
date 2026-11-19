"""Utilidades de visualización para detecciones, atención AIFI/Deformable y distribuciones FDR."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

from seq_transformer.inspector import DeformableCrossCapture, DetectionItem
from seq_transformer.video import FrameMetrics

_PALETTE: list[tuple[int, int, int]] = [
    (239, 68, 68),
    (16, 185, 129),
    (59, 130, 246),
    (245, 158, 11),
    (168, 85, 247),
    (6, 182, 212),
]


def draw_detections(
    image: Image.Image,
    detections: list[DetectionItem],
    *,
    highlight_query_idx: int | None = None,
) -> Image.Image:
    """Dibuja cajas delimitadoras, etiquetas y query ID sobre una copia de la imagen."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for det in detections:
        color = _PALETTE[det.label_id % len(_PALETTE)]
        is_highlighted = highlight_query_idx is not None and det.query_index == highlight_query_idx
        width_px = 4 if is_highlighted else 2
        x1, y1, x2, y2 = det.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width_px)

        caption = f"Q{det.query_index} · {det.label_name} ({det.score:.2f})"
        text_y = max(0.0, y1 - 16.0)
        text_bbox = draw.textbbox((x1, text_y), caption)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1 + 2, text_y), caption, fill=(255, 255, 255))

    return canvas


def plot_encoder_attention_map(
    image: Image.Image,
    encoder_attn: np.ndarray,
    *,
    token_row: int,
    token_col: int,
    head_idx: int | None = None,
) -> Figure:
    """Superpone el mapa de Self-Attention `softmax(QK^T / sqrt(d_k))` del encoder AIFI."""
    # encoder_attn: (num_heads, num_tokens, num_tokens)
    num_tokens = encoder_attn.shape[-1]
    grid_size = math.isqrt(num_tokens)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Se esperaba grilla cuadrada de tokens AIFI, recibido {num_tokens}")

    attn_map = encoder_attn.mean(axis=0) if head_idx is None else encoder_attn[head_idx]
    clamped_row = max(0, min(token_row, grid_size - 1))
    clamped_col = max(0, min(token_col, grid_size - 1))
    token_idx = clamped_row * grid_size + clamped_col

    weights_2d = attn_map[token_idx].reshape(grid_size, grid_size)
    heatmap_pil = Image.fromarray(
        ((weights_2d / (weights_2d.max() + 1e-8)) * 255.0).astype(np.uint8)
    ).resize(image.size, resample=Image.Resampling.BILINEAR)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=120)
    axes[0].imshow(image)
    axes[0].imshow(np.asarray(heatmap_pil), cmap="inferno", alpha=0.55)
    w, h = image.size
    px = (clamped_col + 0.5) * (w / grid_size)
    py = (clamped_row + 0.5) * (h / grid_size)
    axes[0].scatter([px], [py], c="cyan", s=90, marker="x", linewidths=2)
    head_label = "Promedio" if head_idx is None else f"Head {head_idx}"
    axes[0].set_title(f"Self-Attention AIFI ({head_label}) · token ({clamped_row}, {clamped_col})")
    axes[0].axis("off")

    im = axes[1].imshow(weights_2d, cmap="inferno")
    axes[1].set_title(f"Grilla Encoder {grid_size}x{grid_size}")
    axes[1].set_xlabel("Columna espacial")
    axes[1].set_ylabel("Fila espacial")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_deformable_cross_attention(
    image: Image.Image,
    cross_capture: DeformableCrossCapture,
    *,
    query_idx: int,
    head_idx: int | None = None,
    box_xyxy: tuple[float, float, float, float] | None = None,
) -> Figure:
    """Grafica los puntos de muestreo deformable `(x, y)` y pesos de atención de una query."""
    # sampling_locations: (num_queries, n_heads, total_points, 2)
    # attention_weights: (num_queries, n_heads, total_points)
    locs = cross_capture.sampling_locations[query_idx]
    weights = cross_capture.attention_weights[query_idx]

    if head_idx is not None:
        locs = locs[head_idx : head_idx + 1]
        weights = weights[head_idx : head_idx + 1]

    flat_locs = locs.reshape(-1, 2)
    flat_weights = weights.reshape(-1)

    width, height = image.size
    xs = np.clip(flat_locs[:, 0], 0.0, 1.0) * width
    ys = np.clip(flat_locs[:, 1], 0.0, 1.0) * height

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=120)
    ax.imshow(image)

    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="#22d3ee",
            linewidth=2.2,
        )
        ax.add_patch(rect)

    sizes = 35.0 + 480.0 * (flat_weights / (flat_weights.max() + 1e-8))
    scatter = ax.scatter(
        xs,
        ys,
        c=flat_weights,
        s=sizes,
        cmap="plasma",
        alpha=0.88,
        edgecolors="white",
        linewidths=0.6,
    )
    head_label = "todas las cabezas" if head_idx is None else f"Head {head_idx}"
    ax.set_title(f"Cross-Attention Deformable · Query {query_idx} ({head_label})")
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Peso atención")
    fig.tight_layout()
    return fig


def plot_fdr_distributions(fdr_probs: np.ndarray, *, query_idx: int) -> Figure:
    """Grafica las 4 distribuciones discretas de probabilidad FDR para los bordes de la caja."""
    # fdr_probs: (num_queries, 4, max_num_bins + 1)
    edge_probs = fdr_probs[query_idx]
    edge_names = ["Izquierdo (Left)", "Superior (Top)", "Derecho (Right)", "Inferior (Bottom)"]
    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"]
    bins = np.arange(edge_probs.shape[-1])

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.2), dpi=120, sharex=True, sharey=True)
    for idx, ax in enumerate(axes.flat):
        probs = edge_probs[idx]
        ax.bar(bins, probs, color=colors[idx], alpha=0.85, width=0.8)
        expected_bin = float(np.sum(bins * probs))
        ax.axvline(expected_bin, color="#111827", linestyle="--", linewidth=1.4)
        ax.set_title(f"{edge_names[idx]} · E[bin]={expected_bin:.2f}")
        ax.set_xlabel("Bin discreto (0..32)")
        ax.set_ylabel("Probabilidad Pr(n)")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Fine-grained Distribution Refinement (FDR) · Query {query_idx}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_video_metrics(metrics: list[FrameMetrics]) -> Figure:
    """Grafica latencia/FPS y detecciones/confianza a lo largo de la secuencia de video."""
    frame_ids = [m.frame_index for m in metrics]
    latencies = [m.latency_ms for m in metrics]
    detections = [m.num_detections for m in metrics]
    confidences = [m.mean_confidence for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=120)

    axes[0].plot(frame_ids, latencies, marker="o", color="#2563eb", linewidth=2)
    axes[0].set_title("Latencia por frame (ms)")
    axes[0].set_xlabel("Índice de frame")
    axes[0].set_ylabel("ms")
    axes[0].grid(alpha=0.3)

    axes[1].bar(frame_ids, detections, color="#10b981", alpha=0.7, label="Detecciones")
    ax_conf = axes[1].twinx()
    ax_conf.plot(
        frame_ids,
        confidences,
        marker="s",
        color="#f59e0b",
        linewidth=2,
        label="Confianza media",
    )
    ax_conf.set_ylim(0.0, 1.05)
    ax_conf.set_ylabel("Confianza")
    axes[1].set_title("Detecciones y confianza temporal")
    axes[1].set_xlabel("Índice de frame")
    axes[1].set_ylabel("Objetos")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig
