"""
Configuración común de los tests.

Los scripts de src/ no son un paquete instalable: se importan entre sí como
módulos sueltos (`from similarity import ...`). Para que los tests los vean
igual, agregamos src/ al path.

Ningún test necesita audio descargado ni el features.csv real: todo se arma
con señales sintéticas o con DataFrames chicos en un directorio temporal.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventanas: los tests no deben abrir figuras

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
