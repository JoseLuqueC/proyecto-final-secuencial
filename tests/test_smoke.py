"""Proyecto Final Secuencial — prueba de humo."""

from __future__ import annotations

import seq_transformer


def test_el_paquete_expone_una_version() -> None:
    # Arrange / Act
    version = getattr(seq_transformer, "__version__", None)

    # Assert
    assert version is not None
    assert isinstance(version, str)
