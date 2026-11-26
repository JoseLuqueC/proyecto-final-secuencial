# CONSTITUTION — Proyecto Final Secuencial

> **Verdad central del proyecto.** Este documento define el propósito, el stack, la
> estructura, las convenciones y el flujo de trabajo de `Proyecto Final Secuencial`. Si algo aquí
> contradice al código, el código manda **y este archivo se actualiza**.

> ⚠️ **Directiva de mantenimiento:** cada vez que se realicen cambios significativos
> (nuevas funcionalidades, cambios arquitectónicos, adición de clases/módulos o ajustes
> de tecnologías), **se debe actualizar este archivo en el mismo commit** para mantener
> la verdad central del proyecto bien documentada.

---

## 0. Identidad del proyecto

| Campo | Valor |
|-------|-------|
| **Nombre** | Proyecto Final Secuencial |
| **Descripción** | Inferencia, analisis de atencion Q/K/V y visualizacion interactiva de un Transformer Encoder-Decoder en datos secuenciales. |
| **Repositorio** | https://github.com/JoseLuqueC/proyecto-final-secuencial |
| **Autoría** | JoseLuqueC — joseluque93l.lc@gmail.com |
| **Licencia** | MIT |
| **Rama principal** | `main` |
| **Estado** | Funcional en desarrollo 🚀 (D-FINE aprobado) |

## 1. Propósito y no-propósito

### Propósito

Implementar el pipeline de inferencia de una arquitectura **Transformer Encoder–Decoder**
preentrenada (de un artículo científico de los últimos 3 años aprobado por la docente) sobre
datos secuenciales complejos, analizar a profundidad el mecanismo de atención y la generación de
tensores `Q`, `K` y `V`, y demostrar resultados funcionales mediante una interfaz interactiva.

### No-propósito

- Entrenar modelos Transformer desde cero.
- Evaluar arquitecturas que no sean Transformer Encoder–Decoder (p. ej. Decoder-Only o
  Encoder-Only).
- Trabajar sobre series de tiempo tradicionales (datos tabulares, sensores o registros
  estructurados), salvo secuencias de video.

## 2. Alcance

### Dentro del alcance

- **Fase 0 · Base e infraestructura (✅):** Estructura `src-layout`, `uv`, Ruff, `ty`, `pytest`,
  `pre-commit` (incluyendo `detect-secrets`), CI y gobernanza de repositorio.
- **Fase 1 · Selección y aprobación docente (✅):** Aprobado el artículo **D-FINE**
  (*Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*, Peng et al.,
  2024, `arXiv:2410.13842`) sobre secuencias de video.
- **Fase 2 · Inferencia y preprocesamiento (✅):** Carga reproducible de pesos preentrenados
  (`ustc-community/dfine_*_coco` vía `transformers`), extracción de frames y pipeline en video.
- **Fase 3 · Análisis de atención `Q`/`K`/`V` y métricas (✅):** Forward hooks en
  `DFineSelfAttention`, `DFineMultiscaleDeformableAttention` y `DFineIntegral` (FDR), más
  métricas por frame (latencia ms, FPS, detecciones, confianza).
- **Fase 4 · Interfaz interactiva y entrega (🚧):** App en `Streamlit` (`make app`), `README.md`
  con 8 secciones académicas y referencias IEEE, repositorio público en GitHub y capturas
  propias para sustentación (27–28 de noviembre).

### Fuera del alcance (por ahora)

- Entrenamiento o fine-tuning desde cero (el enfoque es comprensión profunda e inferencia).
- Tracking multi-objeto persistente con memoria temporal entre frames (limitación documentada).

## 3. Stack técnico

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje** | Python 3.13+ |
| **Gestor de paquetes y entorno** | [`uv`](https://docs.astral.sh/uv/) |
| **Build backend** | `uv_build` (src-layout) |
| **Paquete Python** | `seq_transformer` |
| **Tipado estático** | [`ty`](https://github.com/astral-sh/ty) |
| **Testing** | [`pytest`](https://docs.pytest.org/) |
| **Linting / Formato** | [Ruff](https://docs.astral.sh/ruff/) |
| **Pre-commit** | [pre-commit](https://pre-commit.com) |
| **CI/CD** | [GitHub Actions](.github/workflows/ci.yml) |
| **Dependencias automáticas** | [Dependabot](.github/dependabot.yml) |
| **Framework Deep Learning** | PyTorch + Hugging Face `transformers` (`DFineForObjectDetection`) |
| **Procesamiento de video** | `opencv-python-headless` + `Pillow` + `NumPy` |
| **Interfaz interactiva y gráficos** | `Streamlit` + `Matplotlib` + `Pandas` |
| **Artículo y modelo base** | D-FINE (`arXiv:2410.13842`, `ustc-community/dfine_n_coco` / `s`/`m`/`l`/`x`) |
| **Seguridad en commits** | `detect-secrets` + `detect-private-key` (`.secrets.baseline`) |

> ⚠️ **Regla dura:** toda gestión de dependencias y entornos pasa por `uv`. Está
> **prohibido** `pip` y `requirements.txt`.

## 4. Estructura del repositorio

```text
proyecto-final-secuencial/
├── src/seq_transformer/
│   ├── __init__.py          # API pública del paquete
│   ├── models.py            # catálogo y carga de pesos D-FINE
│   ├── inspector.py         # forward hooks para Q/K/V, atención y FDR
│   ├── video.py             # muestreo secuencial de video y métricas
│   ├── visualization.py     # render de cajas, heatmaps AIFI/deformable y FDR
│   └── app.py               # interfaz interactiva en Streamlit
├── tests/                   # pruebas offline con fixture DFineConfig mínimo
├── docs/                    # documentación, ADRs e investigación
├── .github/                 # gobernanza (PR/issue templates, CI, Dependabot, labels)
├── .opencode/rules/         # reglas para agentes (tgrep)
├── AGENTS.md                # contexto para agentes LLM
├── CONSTITUTION.md          # este archivo — verdad central
├── README.md                # puerta de entrada pública
├── CONTRIBUTING.md          # guía de contribución
├── CODE_OF_CONDUCT.md       # código de conducta
├── Makefile                 # automatización de tareas
├── pyproject.toml           # metadatos y dependencias (uv)
├── uv.lock                  # lockfile determinista (versionado)
├── .python-version          # versión de Python fijada
└── .pre-commit-config.yaml  # hooks de calidad
```

## 5. Convenciones de código

- **Estilo:** [PEP 8](https://peps.python.org/pep-0008/) y [PEP 257](https://peps.python.org/pep-0257/),
  impuesto por **Ruff** (`line-length = 100`, comillas dobles).
- **Tipado:** anotaciones en funciones y firmas públicas.
- **Nomenclatura:** `snake_case` (funciones/variables), `PascalCase` (clases),
  `UPPER_CASE` (constantes).
- **Docstrings:** en módulos, clases y funciones públicas.
- **Pruebas:** `pytest` con el patrón **AAA** (*Arrange · Act · Assert*).
- **Idioma:** UI, comentarios y commits en español; identificadores de código en inglés.
- **Reglas propias (transversales del proyecto):**
  - La arquitectura seleccionada debe ser estrictamente un **Transformer Encoder–Decoder**
    publicado en los últimos 3 años, con código y pesos preentrenados verificables.
  - Todo el código de inferencia y análisis debe estar claramente comentado en español y ser
    reproducible dentro del tiempo de sustentación (máx. 15 min de exposición).
  - La UI interactiva solo orquesta carga de datos y visualización; la lógica de preprocesamiento,
    inferencia y extracción de tensores `Q`/`K`/`V` vive en `src/seq_transformer/`.
  - No se versionan pesos pesados (`*.pt`, `*.pth`, `*.safetensors`, `*.onnx`), datasets crudos ni
    el PDF privado de especificaciones del curso.

## 6. Calidad y automatización

| Herramienta | Rol | Comando |
|-------------|-----|---------|
| `uv` | Entornos y dependencias | `uv sync` |
| Ruff | Lint + formato | `make lint` / `make format` |
| `ty` | Verificación de tipos | `make typecheck` |
| `pytest` | Pruebas | `make test` |
| `pre-commit` | Hooks antes de cada commit | `make pre-commit` |
| GitHub Actions | CI (pre-commit + pruebas) | `.github/workflows/ci.yml` |

- Ningún commit se salta los hooks (`--no-verify` está prohibido salvo causa justificada y documentada).
- La **CI debe estar en verde** antes de fusionar un Pull Request.
- Toda dependencia nueva entra con `uv add` / `uv add --dev` (nunca editando `uv.lock` a mano).

## 7. Repositorio y Git

- **Remoto:** `https://github.com/JoseLuqueC/proyecto-final-secuencial.git` (o SSH `git@github.com:JoseLuqueC/proyecto-final-secuencial.git`)
- **Rama principal:** `main`
- **Estrategia de ramas:** `feat/...`, `fix/...`, `docs/...`, `refactor/...`, `test/...`, `chore/...`; integración vía Pull Request.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) —
  `feat:`, `fix:`, `docs:`, `refactor:`, `style:`, `test:`, `chore:`.
- **Identidad git:** configurada localmente para `JoseLuqueC` (opcionalmente mediante
  `includeIf` por carpeta en Windows; ver §11).

## 8. Flujo diario (commit + push)

```bash
# 1. Ver cambios
git status

# 2. Verificación completa (lint + formato + pruebas)
make check

# 3. Agregar y commitear (los hooks se ejecutan solos)
git add -A
git commit -m "tipo: mensaje descriptivo"

# 4. Subir a GitHub
git push
```

## 9. Historial de decisiones técnicas

| Fecha | Decisión | Motivo |
|-------|----------|--------|
| 2026-11-14 | Adoptar `uv` como gestor único de entornos y dependencias | Reproducibilidad; prohibido `pip`/`requirements.txt` |
| 2026-11-14 | Adoptar Ruff (lint + formato) y pre-commit | Calidad automatizada antes de cada commit |
| 2026-11-14 | Adoptar Conventional Commits | Historial legible y automatizable |
| 2026-11-14 | Adoptar licencia MIT y `CONSTITUTION.md` como fuente única de verdad | Claridad legal y operativa |
| 2026-11-14 | Mantener artículo, pesos e interfaz en estado pendiente/transversal | Esperar aprobación oficial de la docente antes de acoplar dependencias |
| 2026-11-14 | Fijar `[tool.uv.build-backend] module-name = "seq_transformer"` y `python -m pre_commit` en `Makefile` | Compatibilidad con `uv_build` y política de Control de aplicaciones en Windows (os error 4551 en `pre-commit.exe`) |
| 2026-11-18 | Adoptar D-FINE (`ustc-community/dfine_*_coco`, `attn_implementation="eager"`) + hooks PyTorch | Modelo Encoder-Decoder aprobado (2024); modo `eager` habilita captura real de `encoder_attentions` y `decoder_attentions` |
| 2026-11-18 | Añadir `Yelp/detect-secrets` + `.secrets.baseline` y endurecer `.gitignore` | Prevenir filtración accidental de tokens, `.streamlit/secrets.toml`, cachés HF y videos pesados |
| 2026-11-18 | Construir fixture `DFineConfig` mínimo en `tests/conftest.py` | Ejecutar pruebas reales de hooks `Q/K/V` y `FDR` en CI sin depender de descargas de red |
| 2026-11-18 | Reemplazar `check-added-large-files` por hook Python local y usar `uv run --no-sync` en `Makefile` | Evitar bloqueo `WinError 4551` en binarios cacheados y locks de OneDrive sobre `.venv` |
| 2026-11-24 | Añadir `start_sec` en `video.py` y leer `st.session_state` tras `_render_video_tab` en `app.py` | Permitir saltar a timestamps en videos largos y poblar pestañas 2–4 en el primer clic |
| 2026-11-25 | Separar extras mutuamente excluyentes `cpu` y `cuda` (`pytorch-cu128`) en `pyproject.toml` + `make dev-cuda` | Habilitar GPU NVIDIA local sin penalizar CI ni equipos CPU/macOS |

## 10. Notas y tareas pendientes

- [x] Registrar e implementar modelo aprobado (**D-FINE**, `arXiv:2410.13842`).
- [x] Implementar carga de pesos preentrenados, pipeline de video e inspección de `Q`/`K`/`V`/FDR.
- [x] Construir interfaz interactiva en `Streamlit` (`make app`) y pruebas automatizadas (`pytest`).
- [x] Endurecer `.gitignore` y `pre-commit` con `detect-secrets`.
- [x] Subir repositorio público en GitHub (`JoseLuqueC/proyecto-final-secuencial`).
- [ ] Capturar capturas propias en Streamlit sobre un video real de prueba para la sustentación.
- [ ] Ensayar sustentación (15 min exposición + 5 min preguntas, 27–28 de noviembre).

## 11. Configuración local (no trackear)

> Configuración específica de la máquina local. **NO incluir en commits.**

Si deseas aislar identidad Git por carpeta en Windows (`~/.gitconfig`), puedes aplicar:

```ini
[includeIf "gitdir/i:C:/Users/<usuario>/Ruta/A/Proyectos/"]
    path = ~/.gitconfig-personal
```

`~/.gitconfig-personal`:

```ini
[user]
    name = <Tu Nombre>
    email = <tu-correo>
[core]
    sshCommand = ssh -i C:/Users/<usuario>/.ssh/id_ed25519 -o IdentitiesOnly=yes
```

> Guía completa (macOS/Windows) en el skill `jan-project-standard` → `references/git-identity.md`.
