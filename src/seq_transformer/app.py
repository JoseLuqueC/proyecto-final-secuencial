"""Interfaz interactiva en Streamlit para D-FINE (Transformer Encoder-Decoder en video)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from seq_transformer.inspector import DFineInspector, InspectionResult
from seq_transformer.models import (
    DFINE_MODELS,
    load_dfine_model,
    summarize_architecture,
)
from seq_transformer.video import (
    VideoAnalysisResult,
    analyze_video_sequence,
    create_synthetic_demo_video,
)
from seq_transformer.visualization import (
    draw_detections,
    plot_deformable_cross_attention,
    plot_encoder_attention_map,
    plot_fdr_distributions,
    plot_video_metrics,
)


@st.cache_resource(show_spinner="Cargando pesos preentrenados de D-FINE...")
def _cached_model(model_id: str) -> tuple[Any, Any, str]:
    processor, model, _device = load_dfine_model(model_id, attn_implementation="eager")
    summary = summarize_architecture(model, model_id=model_id)
    return processor, model, summary.device


def _build_synthetic_street_image() -> Image.Image:
    """Genera una escena sintética de fallback si no hay archivo cargado."""
    img = Image.new("RGB", (640, 400), (225, 236, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 210, 640, 400], fill=(85, 90, 98))
    draw.rectangle([80, 235, 220, 310], fill=(35, 100, 210))
    draw.ellipse([98, 298, 126, 326], fill=(20, 20, 20))
    draw.ellipse([174, 298, 202, 326], fill=(20, 20, 20))
    draw.rectangle([390, 175, 425, 285], fill=(210, 65, 65))
    draw.ellipse([394, 145, 421, 172], fill=(240, 195, 160))
    return img


def _render_sidebar() -> tuple[str, float, int, int, bool]:
    st.sidebar.header("Configuración del modelo")
    model_label = st.sidebar.selectbox(
        "Variante D-FINE (COCO)",
        options=list(DFINE_MODELS.keys()),
        index=0,
    )
    model_id = DFINE_MODELS[model_label]
    threshold = st.sidebar.slider("Umbral de confianza", 0.10, 0.90, 0.40, 0.05)
    stride = st.sidebar.slider("Salto entre frames (stride)", 1, 10, 3, 1)
    max_frames = st.sidebar.slider("Máximo de frames a procesar", 4, 40, 12, 2)
    capture_internals = st.sidebar.checkbox(
        "Capturar tensores Q/K/V, atención y FDR",
        value=True,
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Artículo base: Peng et al. (2024), *D-FINE: Redefine Regression Task in DETRs "
        "as Fine-grained Distribution Refinement* (`arXiv:2410.13842`)."
    )
    return model_id, threshold, stride, max_frames, capture_internals


def _render_video_tab(
    inspector: DFineInspector,
    threshold: float,
    stride: int,
    max_frames: int,
    capture_internals: bool,
) -> None:
    st.subheader("Inferencia secuencial en video")
    uploaded_video = st.file_uploader(
        "Sube un video (.mp4, .avi, .mov, .mkv)",
        type=["mp4", "avi", "mov", "mkv"],
    )
    start_sec = float(
        st.number_input(
            "Segundo de inicio en el video (útil para grabaciones largas)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )
    )

    col_a, col_b = st.columns(2)
    run_uploaded = col_a.button("Analizar video subido", disabled=uploaded_video is None)
    run_demo = col_b.button("Generar y analizar video demo sintético")

    if run_uploaded and uploaded_video is not None:
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_video.name).suffix) as tmp:
            tmp.write(uploaded_video.read())
            tmp.flush()
            with st.spinner("Ejecutando inferencia frame a frame..."):
                st.session_state["video_result"] = analyze_video_sequence(
                    Path(tmp.name),
                    inspector,
                    threshold=threshold,
                    stride=stride,
                    max_frames=max_frames,
                    start_sec=start_sec,
                    capture_internals=capture_internals,
                )
    elif run_demo:
        with tempfile.TemporaryDirectory() as tmp_dir:
            demo_path = create_synthetic_demo_video(Path(tmp_dir) / "demo_seq.mp4", num_frames=16)
            with st.spinner("Ejecutando inferencia en secuencia sintética..."):
                st.session_state["video_result"] = analyze_video_sequence(
                    demo_path,
                    inspector,
                    threshold=threshold,
                    stride=1,
                    max_frames=min(max_frames, 16),
                    capture_internals=capture_internals,
                )

    result: VideoAnalysisResult | None = st.session_state.get("video_result")
    if result is None or not result.frames_rgb:
        st.info("Carga un video o genera la secuencia sintética para observar métricas.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames analizados", len(result.frames_rgb))
    m2.metric("Latencia media", f"{result.avg_latency_ms:.1f} ms")
    m3.metric("Throughput medio", f"{result.avg_fps:.2f} FPS")
    m4.metric("Detecciones / frame", f"{result.avg_detections:.2f}")

    st.pyplot(plot_video_metrics(result.metrics))
    plt.close("all")

    selected_idx = st.slider(
        "Explorar frame de la secuencia",
        0,
        len(result.frames_rgb) - 1,
        0,
    )
    frame = result.frames_rgb[selected_idx]
    inspection = result.inspections[selected_idx]
    st.session_state["active_frame"] = frame
    st.session_state["active_inspection"] = inspection

    annotated = draw_detections(frame, inspection.detections)
    st.image(
        annotated,
        caption=f"Frame #{result.metrics[selected_idx].frame_index} · "
        f"{len(inspection.detections)} objetos · {inspection.latency_ms:.1f} ms",
        use_container_width=True,
    )

    if inspection.detections:
        df = pd.DataFrame(
            [
                {
                    "Query ID": d.query_index,
                    "Clase": d.label_name,
                    "Confianza": round(d.score, 4),
                    "Caja [x1, y1, x2, y2]": [round(v, 1) for v in d.box_xyxy],
                }
                for d in inspection.detections
            ]
        )
        st.dataframe(df, use_container_width=True)


def _render_image_inspector_section(
    inspector: DFineInspector,
    threshold: float,
    capture_internals: bool,
) -> tuple[Image.Image | None, InspectionResult | None]:
    with st.expander("O analiza una imagen individual directamente (frame único)", expanded=False):
        uploaded_img = st.file_uploader(
            "Sube una imagen (.jpg, .jpeg, .png, .webp)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        c1, c2 = st.columns(2)
        if c1.button("Inspeccionar imagen subida", disabled=uploaded_img is None) and uploaded_img:
            img = Image.open(uploaded_img).convert("RGB")
            st.session_state["active_frame"] = img
            st.session_state["active_inspection"] = inspector.inspect_image(
                img,
                threshold=threshold,
                capture_internals=capture_internals,
            )
        if c2.button("Inspeccionar imagen sintética"):
            img = _build_synthetic_street_image()
            st.session_state["active_frame"] = img
            st.session_state["active_inspection"] = inspector.inspect_image(
                img,
                threshold=threshold,
                capture_internals=capture_internals,
            )

    return st.session_state.get("active_frame"), st.session_state.get("active_inspection")


def _render_qkv_tab(inspection: InspectionResult | None, arch_summary: Any) -> None:
    st.subheader("Generación y dimensiones de tensores Q, K y V")
    st.markdown(
        f"""
        - **Dimensión oculta (`d_model`):** `{arch_summary.d_model}` ·
          **Cabezas Enc/Dec:** `{arch_summary.encoder_heads}` / `{arch_summary.decoder_heads}` ·
          **`d_k = d_model / heads`:** `{arch_summary.d_model // arch_summary.decoder_heads}`
        - **Self-Attention (`DFineSelfAttention`):** suma `position_embeddings` **solo**
          a `Q` y `K`, no a `V`: `Q = W_q(X + P)`, `K = W_k(X + P)`, `V = W_v(X)`.
        - **Cross-Attention (`DFineMultiscaleDeformableAttention`):** las *Object Queries*
          `Q` proyectan `sampling_offsets` (`dx, dy`) y `attention_weights` sobre la memoria
          del encoder `V = encoder_hidden_states` en `{arch_summary.num_feature_levels}` niveles.
        """
    )

    if inspection is None or not inspection.self_attn_qkv:
        st.info("Ejecuta una inferencia con captura activa para inspeccionar tensores reales.")
        return

    qkv_rows = [
        {
            "Capa": rec.layer_name,
            "Rol": rec.role,
            "Shape Q": str(rec.query.shape),
            "Shape K": str(rec.key.shape),
            "Shape V": str(rec.value.shape),
            "Norma L2 (Q)": round(rec.query.l2_norm, 2),
            "Norma L2 (K)": round(rec.key.l2_norm, 2),
            "Norma L2 (V)": round(rec.value.l2_norm, 2),
        }
        for rec in inspection.self_attn_qkv
    ]
    st.dataframe(pd.DataFrame(qkv_rows), use_container_width=True)

    if inspection.cross_deformable:
        st.markdown("**Capas de Cross-Attention Deformable (Decoder → Encoder):**")
        cross_rows = [
            {
                "Capa": c.layer_name,
                "Query Decoder": str(c.query_input.shape),
                "Memoria Encoder (V)": str(c.value_input.shape),
                "Sampling Offsets": str(c.sampling_offsets.shape),
                "Sampling Locations": str(c.sampling_locations.shape),
                "Attention Weights": str(c.attention_weights.shape),
            }
            for c in inspection.cross_deformable
        ]
        st.dataframe(pd.DataFrame(cross_rows), use_container_width=True)


def _render_attention_tab(frame: Image.Image | None, inspection: InspectionResult | None) -> None:
    st.subheader("Visualización de atención (Encoder Self-Attention & Decoder Cross-Attention)")
    if frame is None or inspection is None:
        st.info("Analiza un video o imagen primero para visualizar mapas de atención.")
        return

    if inspection.encoder_attentions:
        enc_attn = inspection.encoder_attentions[0]
        grid = math.isqrt(enc_attn.shape[-1])
        st.markdown("#### 1. Self-Attention del Encoder (AIFI sobre nivel S5)")
        c1, c2, c3 = st.columns(3)
        row = c1.slider("Fila token origen", 0, grid - 1, grid // 2)
        col = c2.slider("Columna token origen", 0, grid - 1, grid // 2)
        head_opt = c3.selectbox(
            "Cabeza Encoder",
            options=["Promedio"] + [f"Head {i}" for i in range(enc_attn.shape[0])],
        )
        head_idx = None if head_opt == "Promedio" else int(head_opt.split()[-1])
        st.pyplot(
            plot_encoder_attention_map(
                frame,
                enc_attn,
                token_row=row,
                token_col=col,
                head_idx=head_idx,
            )
        )
        plt.close("all")

    if inspection.cross_deformable:
        st.markdown("#### 2. Cross-Attention Deformable del Decoder (Object Query → Imagen)")
        layer_idx = st.slider(
            "Capa del Decoder",
            0,
            len(inspection.cross_deformable) - 1,
            len(inspection.cross_deformable) - 1,
        )
        cross_cap = inspection.cross_deformable[layer_idx]

        det_options = {
            f"Q{d.query_index} · {d.label_name} ({d.score:.2f})": d for d in inspection.detections
        }
        if det_options:
            chosen = st.selectbox("Object Query detectada", options=list(det_options.keys()))
            det = det_options[chosen]
            q_idx = det.query_index
            box = det.box_xyxy
        else:
            q_idx = st.number_input("Índice de Object Query", 0, 299, 0)
            box = None

        head_cross_opt = st.selectbox(
            "Cabeza Cross-Attention",
            options=["Todas"] + [f"Head {i}" for i in range(cross_cap.attention_weights.shape[1])],
        )
        head_cross = None if head_cross_opt == "Todas" else int(head_cross_opt.split()[-1])

        st.pyplot(
            plot_deformable_cross_attention(
                frame,
                cross_cap,
                query_idx=int(q_idx),
                head_idx=head_cross,
                box_xyxy=box,
            )
        )
        plt.close("all")


def _render_fdr_tab(frame: Image.Image | None, inspection: InspectionResult | None) -> None:
    st.subheader("Innovación: Fine-grained Distribution Refinement (FDR)")
    if frame is None or inspection is None or inspection.fdr_edge_probs is None:
        st.info("Ejecuta una inferencia con captura activa para graficar distribuciones FDR.")
        return

    if inspection.detections:
        det_options = {
            f"Q{d.query_index} · {d.label_name} ({d.score:.2f})": d for d in inspection.detections
        }
        chosen = st.selectbox("Objeto detectado", options=list(det_options.keys()), key="fdr_det")
        det = det_options[chosen]
        q_idx = det.query_index
        st.image(
            draw_detections(frame, inspection.detections, highlight_query_idx=q_idx),
            caption=f"Objeto seleccionado: Query {q_idx} ({det.label_name})",
            width=460,
        )
    else:
        q_idx = int(st.number_input("Query ID (sin detecciones sobre umbral)", 0, 299, 0))

    st.pyplot(plot_fdr_distributions(inspection.fdr_edge_probs, query_idx=q_idx))
    plt.close("all")


def _render_theory_tab(arch_summary: Any) -> None:
    st.subheader("Arquitectura, innovaciones y limitaciones de D-FINE")
    st.markdown(
        f"""
        ### 1. Flujo Encoder-Decoder (`{arch_summary.model_id}`)
        1. **Entrada:** Tensor RGB normalizado `(B, 3, H, W)` extraído frame a frame del video.
        2. **Backbone (`HGNetv2`):** Extrae mapas de características multiescala.
        3. **Encoder Híbrido (`DFineHybridEncoder`):**
           - **AIFI (`DFineSelfAttention`):** Self-Attention `softmax(QK^T / sqrt(d_k))V`
             únicamente sobre el mapa semántico `S5` (p. ej., `20x20 = 400` tokens).
           - **CCFF:** Fusión convolucional multiescala (`RepNCSPELAN4`, `SCDown`) que genera la
             memoria del encoder `encoder_hidden_states`.
        4. **Decoder (`DFineDecoder`, `{arch_summary.decoder_layers}` capas):**
           - Selecciona los top-`{arch_summary.num_queries}` **Object Queries** iniciales.
           - **Decoder Self-Attention:** Las queries interactúan entre sí (`Q, K, V`) para evitar
             detecciones duplicadas.
           - **Cross-Attention Deformable:** Cada query predice
             `{sum(arch_summary.decoder_n_points)}` puntos `(dx, dy)` por cabeza
             (`{arch_summary.decoder_heads}` cabezas) sobre la memoria del encoder, combinados
             mediante `DFineGate`.
        5. **Salida:** Logits de `{arch_summary.num_labels}` clases (refinados por **LQE**) y cajas
           integradas desde distribuciones de `{arch_summary.max_num_bins + 1}` bins (**FDR**).

        ### 2. Innovaciones frente a DETR clásico
        - **FDR (Fine-grained Distribution Refinement):** Modela cada borde de la caja como una
          distribución discreta refinada capa por capa, capturando incertidumbre real.
        - **GO-LSD (Global Optimal Localization Self-Distillation):** Destila conocimiento desde las
          distribuciones más finas de la última capa hacia capas tempranas en entrenamiento.
        - **LQE (Localization Quality Estimation):** Ajusta la confianza de clasificación usando los
          valores `top-k` de las distribuciones FDR.

        ### 3. Limitaciones identificadas
        - **Sin memoria temporal entre frames:** Procesa cada frame de forma independiente;
          no propaga estados ocultos ni identidades entre `t-1` y `t`.
        - **Dependencia de vocabulario cerrado:** Preentrenado en 80 clases de COCO (no detecta
          categorías fuera de COCO sin *fine-tuning*).
        - **Interpretabilidad en Cross-Attention:** Usa atención deformable dispersa en lugar de
          matriz densa `Q x K^T` completa, requiriendo visualizar `sampling_locations`.
        """
    )


def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    st.set_page_config(
        page_title="D-FINE · Transformer Encoder-Decoder en Video",
        layout="wide",
    )
    st.title("D-FINE: Inferencia y Análisis Q/K/V en Secuencias de Video")
    st.caption(
        "Proyecto Final · Procesamiento de Datos Secuenciales (UAO) · "
        "Transformer Encoder-Decoder (`arXiv:2410.13842`)"
    )

    model_id, threshold, stride, max_frames, capture_internals = _render_sidebar()
    processor, model, device_str = _cached_model(model_id)
    arch_summary = summarize_architecture(model, model_id=model_id)
    inspector = DFineInspector(model=model, processor=processor)

    st.sidebar.success(
        f"Modelo listo en `{device_str}` · {arch_summary.total_params / 1e6:.2f}M parámetros"
    )

    _render_image_inspector_section(inspector, threshold, capture_internals)

    tab_video, tab_qkv, tab_attn, tab_fdr, tab_theory = st.tabs(
        [
            "1. Video e Inferencia",
            "2. Tensores Q / K / V",
            "3. Mapas de Atención",
            "4. Innovación FDR",
            "5. Arquitectura y Limitaciones",
        ]
    )

    with tab_video:
        _render_video_tab(inspector, threshold, stride, max_frames, capture_internals)

    frame = st.session_state.get("active_frame")
    inspection = st.session_state.get("active_inspection")

    with tab_qkv:
        _render_qkv_tab(inspection, arch_summary)
    with tab_attn:
        _render_attention_tab(frame, inspection)
    with tab_fdr:
        _render_fdr_tab(frame, inspection)
    with tab_theory:
        _render_theory_tab(arch_summary)


if __name__ == "__main__":
    main()
