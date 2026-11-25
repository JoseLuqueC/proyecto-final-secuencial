"""Pruebas de regresión del muestreo temporal de videos D-FINE."""

from pathlib import Path

import pytest

from seq_transformer.video import create_synthetic_demo_video, extract_video_frames


@pytest.fixture
def short_video(tmp_path: Path) -> Path:
    """Crea un video reproducible de 16 frames a 8 FPS (2 segundos)."""
    return create_synthetic_demo_video(
        tmp_path / "short_video.mp4",
        num_frames=16,
        width=320,
        height=180,
        fps=8.0,
    )


def test_start_sec_zero_extrae_desde_el_primer_frame(short_video: Path) -> None:
    frames, source_fps = extract_video_frames(
        short_video,
        stride=3,
        max_frames=4,
        start_sec=0.0,
    )

    assert source_fps == pytest.approx(8.0)
    assert [frame_idx for frame_idx, _, _ in frames] == [0, 3, 6, 9]
    assert all(image.size == (320, 180) for _, _, image in frames)


def test_start_sec_un_segundo_conserva_indices_y_tiempos(short_video: Path) -> None:
    frames, _ = extract_video_frames(
        short_video,
        stride=2,
        max_frames=4,
        start_sec=1.0,
    )

    assert [frame_idx for frame_idx, _, _ in frames] == [8, 10, 12, 14]
    assert [timestamp for _, timestamp, _ in frames] == pytest.approx([1.0, 1.25, 1.5, 1.75])


def test_inicio_posterior_a_duracion_no_devuelve_frames(short_video: Path) -> None:
    frames, source_fps = extract_video_frames(
        short_video,
        stride=1,
        max_frames=4,
        start_sec=10.0,
    )

    assert source_fps == pytest.approx(8.0)
    assert frames == []


@pytest.mark.parametrize(
    ("stride", "max_frames", "start_sec", "mensaje"),
    [
        (0, 4, 0.0, "stride"),
        (1, 0, 0.0, "max_frames"),
        (1, 4, -0.1, "start_sec"),
    ],
)
def test_rechaza_parametros_invalidos(
    short_video: Path,
    stride: int,
    max_frames: int,
    start_sec: float,
    mensaje: str,
) -> None:
    with pytest.raises(ValueError, match=mensaje):
        extract_video_frames(
            short_video,
            stride=stride,
            max_frames=max_frames,
            start_sec=start_sec,
        )


def test_video_inexistente_informa_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No se pudo abrir el video"):
        extract_video_frames(tmp_path / "inexistente.mp4")
