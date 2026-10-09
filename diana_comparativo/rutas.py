"""Ubicación de los datos locales autorizados.

Los datos de los participantes (poses, secuencias, manifiestos y resultados
derivados) no forman parte de este repositorio. Cada usuario autorizado crea
``configuracion/rutas.json`` a partir de ``configuracion/rutas.ejemplo.json``.

Las rutas relativas se resuelven contra ``datos_raiz``; ``salida`` debe ser una
carpeta fuera del repositorio.
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[1]
RUTAS_POR_DEFECTO = RAIZ_REPOSITORIO / "configuracion" / "rutas.json"


class Rutas:
    def __init__(self, archivo: Path | str = RUTAS_POR_DEFECTO):
        archivo = Path(archivo)
        if not archivo.exists():
            raise SystemExit(
                f"No existe {archivo}. Copie configuracion/rutas.ejemplo.json como rutas.json "
                "e indique la ubicación de los datos locales autorizados."
            )
        self.valores = json.loads(archivo.read_text(encoding="utf-8"))
        self.datos_raiz = Path(self.valores["datos_raiz"]).expanduser()
        self.salida = Path(self.valores["salida"]).expanduser()
        if RAIZ_REPOSITORIO in self.salida.resolve().parents or self.salida.resolve() == RAIZ_REPOSITORIO:
            raise SystemExit("La carpeta de salida debe estar fuera del repositorio (contiene datos derivados).")

    def __getitem__(self, clave: str) -> Path:
        ruta = Path(self.valores[clave]).expanduser()
        return ruta if ruta.is_absolute() else self.datos_raiz / ruta

    def salida_de(self, *partes: str) -> Path:
        ruta = self.salida.joinpath(*partes)
        ruta.mkdir(parents=True, exist_ok=True)
        return ruta
