#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREPARAR DATASETS PÚBLICOS PARA EL EXPERIMENTO 1
=================================================
Convierte datasets públicos al formato que espera `exp1_linea_base_textual.py`:

    llamada_id | texto | etiqueta (1 = enfado/riesgo, 0 = no) | [cliente_id]

IMPORTANTE: ningún dataset público tiene exactamente "llamadas de clientes con
riesgo de baja". Aquí se usan PROXIES (sustitutos razonables):

  --fuente emoevent  Tweets en ESPAÑOL con emoción etiquetada (Ekman).
                     Proxy de enfado:  etiqueta = 1 si la emoción es 'anger'.
                     Es la prueba más realista de "detectar enfado" en español,
                     aunque son tweets, no llamadas.
                     Requiere cuenta de Hugging Face y aceptar las condiciones
                     del dataset (acceso con contacto): `huggingface-cli login`.

  --fuente bitext    Consultas de atención al cliente (INGLÉS) con intención.
                     Proxy de riesgo de baja: intents de queja/cancelación/cambio
                     de cuenta. Es texto sintético y muy limpio: A sacará
                     métricas altísimas. Útil para probar la tubería, no para
                     sacar conclusiones.

  --fuente iemocap   IEMOCAP (inglés, ACTUADO, ~12 h, audio + transcripción + emoción).
                     Requiere haber obtenido el dataset con licencia académica (USC SAIL)
                     y descomprimido la carpeta `IEMOCAP_full_release`:
                         --ruta /ruta/a/IEMOCAP_full_release
                     Proxy de enfado: etiqueta = 1 si la emoción es 'ang' o 'fru'
                     (enfadado / frustrado). Cambiable con --positivos.
                     Añade `audio_path` y `cliente_id` (= sesión) para reutilizarlo
                     en el Experimento 2 (audio).

  --fuente csv       Cualquier CSV propio o descargado (Kaggle, etc.).
                     Indica --col-texto, --col-etiqueta y --positivos.

Ejemplos:
    python preparar_dataset_publico.py --fuente emoevent --out llamadas.csv
    python preparar_dataset_publico.py --fuente bitext   --out llamadas.csv
    python preparar_dataset_publico.py --fuente iemocap --ruta IEMOCAP_full_release --out llamadas.csv
    python preparar_dataset_publico.py --fuente csv --ruta quejas.csv \\
        --col-texto narrative --col-etiqueta product --positivos "Mortgage" --out llamadas.csv

Dependencias: pandas  (y `datasets` para emoevent/bitext:  pip install datasets)
"""
from __future__ import annotations

import argparse

import pandas as pd

# Intents de Bitext que usamos como proxy de "enfado / riesgo de baja"
BITEXT_INTENTS_RIESGO = {
    "complaint",                # queja directa
    "delete_account",           # quiere borrar su cuenta  (~ baja)
    "switch_account",           # cambiar/migrar de cuenta (~ portabilidad)
    "check_cancellation_fee",   # pregunta cuánto cuesta cancelar (~ intención de irse)
}


# ----------------------------------------------------------------------------
# Conversión común al formato del experimento
# ----------------------------------------------------------------------------
def a_formato(df: pd.DataFrame, col_texto: str, col_etiqueta: str, positivos,
              col_grupo: str | None = None, max_n: int | None = None,
              seed: int = 42, dedup: bool = True, min_len: int = 5,
              extras: list | None = None) -> pd.DataFrame:
    """Devuelve un DataFrame con las columnas que lee exp1_linea_base_textual.py."""
    for c in (col_texto, col_etiqueta):
        if c not in df.columns:
            raise ValueError(f"No existe la columna '{c}'. Columnas disponibles: {list(df.columns)}")

    pos = {str(p).strip().lower() for p in positivos}
    out = pd.DataFrame({
        "texto": df[col_texto].fillna("").astype(str).str.strip(),
        "etiqueta": df[col_etiqueta].astype(str).str.strip().str.lower().isin(pos).astype(int),
    })
    if col_grupo:
        out["cliente_id"] = df[col_grupo].astype(str)

    for c in (extras or []):          # columnas que se arrastran tal cual (p. ej. audio_path)
        out[c] = df[c].values

    # Limpieza: textos vacíos/muy cortos y duplicados exactos (evitan fuga train/test)
    out = out[out["texto"].str.len() >= min_len]
    if dedup:
        out = out.drop_duplicates(subset="texto")

    if max_n and len(out) > max_n:
        out = out.sample(max_n, random_state=seed)

    out = out.reset_index(drop=True)
    out.insert(0, "llamada_id", [f"d{i:06d}" for i in range(len(out))])

    tasa = out["etiqueta"].mean()
    print(f"Filas: {len(out)} | positivos (etiqueta=1): {out['etiqueta'].sum()} ({tasa:.1%})")
    if tasa < 0.01 or tasa > 0.5:
        print("[aviso] Tasa de positivos extrema: revisa --positivos o el mapeo de etiquetas.")
    return out


# ----------------------------------------------------------------------------
# Cargadores
# ----------------------------------------------------------------------------
def _concat_splits(ds_dict) -> pd.DataFrame:
    """Une train/validation/test: el experimento hace su propio split reproducible.
    Si una columna es ClassLabel (entero), la convierte a su nombre de texto."""
    frames = []
    for name, split in ds_dict.items():
        df = split.to_pandas()
        for col, feat in split.features.items():
            if hasattr(feat, "int2str") and col in df.columns:
                df[col] = df[col].map(lambda i, f=feat: f.int2str(int(i)))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def cargar_emoevent(positivos, max_n):
    from datasets import get_dataset_config_names, load_dataset

    repo = "fmplaza/EmoEvent"
    try:
        configs = get_dataset_config_names(repo)
    except Exception as e:  # dataset con acceso condicionado: falta login/aceptar condiciones
        raise SystemExit(
            f"No se pudo acceder a {repo}: {e}\n"
            "-> Crea cuenta en huggingface.co, abre https://huggingface.co/datasets/fmplaza/EmoEvent, "
            "acepta las condiciones y ejecuta `huggingface-cli login` (o `hf auth login`)."
        )
    print("Configuraciones disponibles:", configs)
    cfg_es = next((c for c in configs if c.lower().endswith("es")), None)
    if cfg_es is None:
        raise SystemExit(f"No encuentro la versión en español entre: {configs}")
    ds = load_dataset(repo, cfg_es)
    df = _concat_splits(ds)

    col_texto = next((c for c in ("tweet", "text", "texto") if c in df.columns), None)
    col_lab = next((c for c in ("emotion", "label") if c in df.columns), None)
    if not col_texto or not col_lab:
        raise SystemExit(f"Columnas inesperadas: {list(df.columns)}. Usa --fuente csv.")
    print("Distribución de emociones:\n", df[col_lab].value_counts().to_string())
    return a_formato(df, col_texto, col_lab, positivos or ["anger"], max_n=max_n)


def cargar_bitext(positivos, max_n):
    from datasets import load_dataset

    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = _concat_splits(ds)
    col_texto = "instruction" if "instruction" in df.columns else "utterance"
    return a_formato(df, col_texto, "intent", positivos or BITEXT_INTENTS_RIESGO, max_n=max_n)


def cargar_iemocap(ruta, positivos, max_n):
    """Lee IEMOCAP_full_release: transcripciones + etiquetas categóricas por enunciado."""
    import re
    from pathlib import Path

    raiz = Path(ruta)
    if not raiz.exists():
        raise SystemExit(f"No existe la carpeta {ruta}. Indica --ruta .../IEMOCAP_full_release")

    # Línea de transcripción:  Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
    re_trans = re.compile(r"^(\S+)\s+\[[^\]]*\]:\s*(.*)$")
    # Línea de etiqueta:       [6.2901 - 8.2357]<TAB>Ses01F_impro01_F000<TAB>neu<TAB>[2.5, 2.5, 2.5]
    re_emo = re.compile(r"^\[[\d\.]+ - [\d\.]+\]\t(\S+)\t(\w+)\t")

    textos, emociones = {}, {}
    for ses in sorted(raiz.glob("Session*")):
        for f in (ses / "dialog" / "transcriptions").glob("*.txt"):
            for linea in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = re_trans.match(linea.strip())
                if m and m.group(1).startswith("Ses"):
                    # quitar marcas no verbales: [LAUGHTER], [GARBAGE], [BREATHING]...
                    limpio = re.sub(r"\[[^\]]*\]", " ", m.group(2))
                    textos[m.group(1)] = " ".join(limpio.split())
        for f in (ses / "dialog" / "EmoEvaluation").glob("*.txt"):
            for linea in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = re_emo.match(linea)
                if m:
                    emociones[m.group(1)] = m.group(2)

    ids = sorted(set(textos) & set(emociones))
    if not ids:
        raise SystemExit("No se encontraron enunciados. ¿Es la carpeta IEMOCAP_full_release correcta?")

    df = pd.DataFrame({
        "utt": ids,
        "texto_raw": [textos[i] for i in ids],
        "emocion": [emociones[i] for i in ids],
    })
    df["sesion"] = "Session" + df["utt"].str[3:5].astype(int).astype(str)
    df["audio_path"] = df.apply(
        lambda r: f"{r['sesion']}/sentences/wav/{r['utt'].rsplit('_', 1)[0]}/{r['utt']}.wav", axis=1)

    # 'xxx' = los anotadores no se pusieron de acuerdo; 'oth' = otra emoción: se descartan
    df = df[~df["emocion"].isin(["xxx", "oth"])]
    print("Distribución de emociones:\n", df["emocion"].value_counts().to_string())
    return a_formato(df, "texto_raw", "emocion", positivos or ["ang", "fru"],
                     col_grupo="sesion", max_n=max_n, dedup=False, min_len=2,
                     extras=["audio_path"])


def cargar_csv(args):
    df = pd.read_csv(args.ruta)
    if not (args.col_texto and args.col_etiqueta and args.positivos):
        raise SystemExit("Con --fuente csv indica --ruta, --col-texto, --col-etiqueta y --positivos.")
    return a_formato(df, args.col_texto, args.col_etiqueta, args.positivos,
                     col_grupo=args.col_grupo, max_n=args.max_n)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Prepara datasets públicos para el Experimento 1")
    ap.add_argument("--fuente", required=True, choices=["emoevent", "bitext", "iemocap", "csv"])
    ap.add_argument("--out", default="llamadas.csv")
    ap.add_argument("--max-n", type=int, default=None, help="limitar nº de filas (muestreo aleatorio)")
    ap.add_argument("--positivos", nargs="+", default=None,
                    help="valores de la columna de etiqueta que cuentan como 1 "
                         "(emoevent: emociones, p.ej. anger disgust; bitext: intents)")
    # Solo para --fuente csv
    ap.add_argument("--ruta", help="ruta del CSV (csv) o de la carpeta IEMOCAP_full_release (iemocap)")
    ap.add_argument("--col-texto")
    ap.add_argument("--col-etiqueta")
    ap.add_argument("--col-grupo", default=None, help="columna que identifica al cliente (opcional)")
    args = ap.parse_args()

    if args.fuente == "emoevent":
        df = cargar_emoevent(args.positivos, args.max_n)
    elif args.fuente == "bitext":
        df = cargar_bitext(args.positivos, args.max_n)
    elif args.fuente == "iemocap":
        df = cargar_iemocap(args.ruta, args.positivos, args.max_n)
    else:
        df = cargar_csv(args)

    df.to_csv(args.out, index=False)
    print(f"Guardado: {args.out}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
