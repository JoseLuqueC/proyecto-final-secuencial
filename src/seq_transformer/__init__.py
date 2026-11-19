"""Paquete principal para inferencia e inspección Q/K/V de D-FINE en video."""

from seq_transformer.inspector import (
    DeformableCrossCapture,
    DetectionItem,
    DFineInspector,
    InspectionResult,
    SelfAttentionQKV,
    TensorStats,
)
from seq_transformer.models import (
    DFINE_MODELS,
    DFineArchitectureSummary,
    load_dfine_model,
    resolve_device,
    summarize_architecture,
)
from seq_transformer.video import (
    FrameMetrics,
    VideoAnalysisResult,
    analyze_video_sequence,
    create_synthetic_demo_video,
    extract_video_frames,
)

__version__ = "0.1.0"

__all__ = [
    "DFINE_MODELS",
    "DFineArchitectureSummary",
    "DFineInspector",
    "DeformableCrossCapture",
    "DetectionItem",
    "FrameMetrics",
    "InspectionResult",
    "SelfAttentionQKV",
    "TensorStats",
    "VideoAnalysisResult",
    "__version__",
    "analyze_video_sequence",
    "create_synthetic_demo_video",
    "extract_video_frames",
    "load_dfine_model",
    "resolve_device",
    "summarize_architecture",
]
