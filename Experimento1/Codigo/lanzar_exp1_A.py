#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LANZADOR — EXPERIMENTO 1, PARTE A (TF-IDF + Regresión Logística) CON EMOEVENT
==============================================================================
Hace todo el recorrido en un solo comando:

    0. Comprueba que están instaladas las librerías necesarias.
    1. Comprueba que has iniciado sesión en Hugging Face (EmoEvent es de acceso
       condicionado).
    2. Descarga EmoEvent (español) y lo convierte al formato del experimento
       (etiqueta = 1 si la emoción es 'anger').
    3. Ejecuta el candidato A.
    4. Muestra un resumen con comprobaciones de sentido común.

Requisitos previos (una sola vez):
    pip install numpy pandas scikit-learn datasets huggingface_hub
    # Aceptar condiciones en https://huggingface.co/datasets/fmplaza/EmoEvent
    huggingface-cli login        # (o `hf auth login`)

Uso:
    python lanzar_exp1_A.py                          # EmoEvent, anger = 1
    python lanzar_exp1_A.py --positivos anger disgust
    python lanzar_exp1_A.py --fuente bitext          # alternativa sin cuenta (inglés)
    python lanzar_exp1_A.py --csv mis_datos.csv      # saltar descarga: usar un CSV ya preparado

Este fichero debe estar en la MISMA carpeta que:
    preparar_dataset_publico.py   y   exp1_linea_base_textual.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PREPARAR = AQUI / "preparar_dataset_publico.py"
EXP1 = AQUI / "exp1_linea_base_textual.py"

# módulo importable -> nombre del paquete en pip
DEPENDENCIAS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
}
DEPENDENCIAS_DESCARGA = {
    "datasets": "datasets",
    "huggingface_hub": "huggingface_hub",
}


def titulo(texto: str) -> None:
    print("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)


def comprobar_dependencias(necesita_descarga: bool) -> None:
    """Paso 0: avisa de lo que falta con el comando exacto de pip."""
    requeridas = dict(DEPENDENCIAS)
    if necesita_descarga:
        requeridas.update(DEPENDENCIAS_DESCARGA)
    faltan = [pip for mod, pip in requeridas.items() if importlib.util.find_spec(mod) is None]
    if faltan:
        sys.exit(f"Faltan librerías. Instálalas con:\n    pip install {' '.join(faltan)}")
    print("Dependencias: OK")


def comprobar_login_hf() -> None:
    """Paso 1: EmoEvent requiere sesión iniciada y condiciones aceptadas."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            try:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
            except Exception:
                token = None
    if not token:
        sys.exit(
            "No hay sesión de Hugging Face. Haz esto una vez:\n"
            "  1) Crea cuenta en https://huggingface.co y acepta las condiciones en\n"
            "     https://huggingface.co/datasets/fmplaza/EmoEvent\n"
            "  2) Ejecuta:  huggingface-cli login   (o  hf auth login)\n"
            "     o define la variable de entorno HF_TOKEN.\n"
            "Alternativa sin cuenta:  python lanzar_exp1_A.py --fuente bitext"
        )
    print("Sesión de Hugging Face: OK")


def ejecutar(cmd: list) -> None:
    """Lanza un subproceso mostrando su salida en vivo; aborta si falla."""
    print("$", " ".join(str(c) for c in cmd), "\n")
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        sys.exit(f"\nEl paso anterior falló (código {r.returncode}). Revisa el mensaje de arriba.")


def resumen(out_dir: Path, csv_path: Path) -> None:
    """Paso 4: resultados + comprobaciones de sentido común."""
    import pandas as pd

    metricas = pd.read_csv(out_dir / "metricas.csv")
    fila = metricas[metricas["modelo"] == "A"].iloc[0]
    datos = pd.read_csv(csv_path)
    prevalencia = float(datos["etiqueta"].mean())
    extra = json.loads((out_dir / "metricas.json").read_text())["A"]["extra"]

    titulo("RESUMEN DEL CANDIDATO A")
    print(f"Filas totales: {len(datos)} | prevalencia de 'enfado': {prevalencia:.1%}")
    print(f"F1        : {fila['f1']:.3f}  IC95% {fila['f1_ic95']}")
    print(f"PR-AUC    : {fila['pr_auc']:.3f}  IC95% {fila['pr_auc_ic95']}  "
          f"(azar ≈ {prevalencia:.3f})")
    print(f"Precisión : {fila['precision']:.3f} | Recall: {fila['recall']:.3f} | "
          f"ROC-AUC: {fila['roc_auc']:.3f}")
    print(f"Coste     : entrenamiento {fila['train_s']:.1f} s | "
          f"latencia p50 {fila['lat_ms_p50']:.2f} ms | modelo {fila['size_mb']:.2f} MB")
    print(f"C elegido : {extra['C_elegido']}")
    print("Palabras que más empujan a 'enfado':", ", ".join(extra["top_palabras_riesgo"]))
    print("Palabras que más empujan a 'no enfado':", ", ".join(extra["top_palabras_no_riesgo"]))

    titulo("COMPROBACIONES DE SENTIDO COMÚN")
    avisos = []
    if fila["pr_auc"] < prevalencia * 1.3:
        avisos.append("PR-AUC apenas supera al azar: el texto aporta poca señal o hay un "
                      "problema con las etiquetas.")
    if fila["pr_auc"] > 0.97:
        avisos.append("PR-AUC casi perfecto: sospecha de fuga de datos o de un dataset "
                      "demasiado fácil (plantillas, duplicados).")
    if prevalencia < 0.05:
        avisos.append("Muy pocos positivos: los intervalos serán anchos; prueba con "
                      "--positivos anger disgust.")
    if fila["recall"] < 0.4:
        avisos.append("Recall bajo: A deja pasar muchos enfados. Es lo esperable de un "
                      "modelo que solo cuenta palabras.")
    if not avisos:
        avisos.append("Sin alertas: el resultado es coherente. Guárdalo como línea base y "
                      "pasa a B y C.")
    for a in avisos:
        print(" -", a)

    print(f"\nTodo guardado en: {out_dir.resolve()}")
    print("Siguiente paso: reutiliza el mismo CSV y split para B y C:")
    print(f"    python exp1_linea_base_textual.py --data {csv_path} --models A B C "
          f"--idioma es --out resultados_completos")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lanza el Experimento 1 parte A")
    ap.add_argument("--fuente", choices=["emoevent", "bitext"], default="emoevent")
    ap.add_argument("--positivos", nargs="+", default=None,
                    help="emociones/intents que cuentan como 1 (EmoEvent: anger disgust ...)")
    ap.add_argument("--csv", default=None, help="usar un CSV ya preparado (salta la descarga)")
    ap.add_argument("--max-n", type=int, default=None, help="limitar nº de filas")
    ap.add_argument("--out-dir", default="resultados_exp1_A")
    ap.add_argument("--datos", default="llamadas.csv", help="CSV intermedio que se genera")
    ap.add_argument("--forzar", action="store_true", help="volver a descargar aunque exista el CSV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    for f in (PREPARAR, EXP1):
        if not f.exists():
            sys.exit(f"No encuentro {f.name} en {AQUI}. Deja los tres scripts en la misma carpeta.")

    csv_path = Path(args.csv) if args.csv else Path(args.datos)
    hay_que_preparar = not args.csv and (args.forzar or not csv_path.exists())

    titulo("PASO 0 — Dependencias")
    comprobar_dependencias(necesita_descarga=hay_que_preparar)

    titulo("PASO 1 — Datos")
    if hay_que_preparar:
        if args.fuente == "emoevent":
            comprobar_login_hf()
        cmd = [sys.executable, PREPARAR, "--fuente", args.fuente, "--out", csv_path]
        if args.positivos:
            cmd += ["--positivos", *args.positivos]
        if args.max_n:
            cmd += ["--max-n", args.max_n]
        ejecutar(cmd)
    else:
        print(f"Usando CSV existente: {csv_path}  (usa --forzar para volver a descargar)")

    if not csv_path.exists():
        sys.exit(f"No existe {csv_path}.")

    titulo("PASO 2 — Candidato A (TF-IDF + Regresión Logística)")
    idioma = "es" if args.fuente == "emoevent" or args.csv else "en"
    ejecutar([sys.executable, EXP1, "--data", csv_path, "--models", "A",
              "--idioma", idioma, "--out", args.out_dir,
              "--seed", args.seed, "--bootstrap", args.bootstrap])

    resumen(Path(args.out_dir), csv_path)


if __name__ == "__main__":
    main()
