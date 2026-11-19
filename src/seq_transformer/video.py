"""Pipeline de procesamiento secuencial sobre video y métricas por frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from seq_transformer.inspector import DFineInspector, InspectionResult


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """Métricas de inferencia recolectadas para un frame individual de la secuencia."""

    frame_index: int
    timestamp_sec: float
    latency_ms: float
    fps: float
    num_detections: int
    mean_confidence: float


@dataclass(slots=True)
class VideoAnalysisResult:
    """Resultado completo de procesar una secuencia de video con D-FINE."""

    frames_rgb: list[Image.Image] = field(default_factory=list)
    inspections: list[InspectionResult] = field(default_factory=list)
    metrics: list[FrameMetrics] = field(default_factory=list)
    source_fps: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Latencia promedio por frame en milisegundos."""
        if not self.metrics:
            return 0.0
        return float(np.mean([m.latency_ms for m in self.metrics]))

    @property
    def avg_fps(self) -> float:
        """Rendimiento promedio en frames por segundo."""
        if not self.metrics:
            return 0.0
        return float(np.mean([m.fps for m in self.metrics]))

    @property
    def avg_detections(self) -> float:
        """Promedio de objetos detectados por frame."""
        if not self.metrics:
            return 0.0
        return float(np.mean([m.num_detections for m in self.metrics]))


def create_synthetic_demo_video(
    output_path: Path,
    *,
    num_frames: int = 16,
    width: int = 640,
    height: int = 360,
    fps: float = 8.0,
) -> Path:
    """Genera un video sintético MP4 liviano para pruebas y demostración offline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo inicializar VideoWriter en {output_path}")

    try:
        for idx in range(num_frames):
            frame = np.full((height, width, 3), (232, 238, 245), dtype=np.uint8)
            # Fondo tipo vía urbana simple para simular desplazamiento secuencial
            cv2.rectangle(frame, (0, height // 2), (width, height), (95, 95, 105), -1)
            cv2.line(
                frame,
                (0, int(height * 0.75)),
                (width, int(height * 0.75)),
                (230, 230, 230),
                3,
            )

            cx = int(70 + (width - 140) * (idx / max(num_frames - 1, 1)))
            cy = int(height * 0.65)
            cv2.rectangle(frame, (cx - 45, cy - 25), (cx + 45, cy + 25), (30, 90, 200), -1)
            cv2.circle(frame, (cx - 25, cy + 28), 10, (25, 25, 25), -1)
            cv2.circle(frame, (cx + 25, cy + 28), 10, (25, 25, 25), -1)
            cv2.putText(
                frame,
                f"Frame {idx + 1:02d}/{num_frames}",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 40),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def extract_video_frames(
    video_path: Path,
    *,
    stride: int = 2,
    max_frames: int = 24,
) -> tuple[list[tuple[int, float, Image.Image]], float]:
    """Extrae frames RGB muestreados de un video preservando índice y timestamp."""
    if stride < 1:
        raise ValueError("stride debe ser >= 1")
    if max_frames < 1:
        raise ValueError("max_frames debe ser >= 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_fps = raw_fps if raw_fps > 0.0 else 24.0
    sampled: list[tuple[int, float, Image.Image]] = []
    frame_idx = 0

    try:
        while len(sampled) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                sampled.append((frame_idx, frame_idx / source_fps, Image.fromarray(rgb)))
            frame_idx += 1
    finally:
        cap.release()

    return sampled, source_fps


def analyze_video_sequence(
    video_path: Path,
    inspector: DFineInspector,
    *,
    threshold: float = 0.4,
    stride: int = 3,
    max_frames: int = 20,
    capture_internals: bool = True,
) -> VideoAnalysisResult:
    """Ejecuta inferencia D-FINE frame a frame y agrega métricas secuenciales."""
    sampled, source_fps = extract_video_frames(
        video_path,
        stride=stride,
        max_frames=max_frames,
    )

    result = VideoAnalysisResult(source_fps=source_fps)
    for frame_idx, timestamp_sec, pil_frame in sampled:
        inspection = inspector.inspect_image(
            pil_frame,
            threshold=threshold,
            capture_internals=capture_internals,
        )
        fps_instant = 1000.0 / inspection.latency_ms if inspection.latency_ms > 0.0 else 0.0
        mean_conf = (
            float(np.mean([d.score for d in inspection.detections]))
            if inspection.detections
            else 0.0
        )

        result.frames_rgb.append(pil_frame)
        result.inspections.append(inspection)
        result.metrics.append(
            FrameMetrics(
                frame_index=frame_idx,
                timestamp_sec=timestamp_sec,
                latency_ms=inspection.latency_ms,
                fps=fps_instant,
                num_detections=len(inspection.detections),
                mean_confidence=mean_conf,
            )
        )

    return result
