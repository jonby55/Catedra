#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENTO 1 — LÍNEA BASE TEXTUAL
===================================
Objetivo: comparar tres formas de clasificar el texto transcrito de una llamada
(¿cliente enfadado / con riesgo de baja?) midiendo CALIDAD y COSTE, para decidir
si merece la pena complicarse con modelos más caros.

    A) TF-IDF + Regresión Logística      -> barato, ultrarrápido, "cuenta palabras"
    B) Transformer ajustado (fine-tuning) -> caro, entiende contexto
    C) Embeddings congelados + Boosting   -> punto medio (+ variante con datos de ficha)

Formato del CSV de entrada (--data):
    llamada_id  : identificador único de la llamada
    cliente_id  : (recomendado) un mismo cliente puede tener varias llamadas
    texto       : transcripción de la llamada
    etiqueta    : 1 = enfadado / riesgo de baja, 0 = no
    (opcionales) antiguedad_meses, factura_media, n_reclamaciones_12m ... (ficha del cliente)

Uso rápido:
    python exp1_linea_base_textual.py --demo --models A          # prueba sin datos reales
    python exp1_linea_base_textual.py --data llamadas.csv --models A B C

Dependencias:
    Siempre : numpy pandas scikit-learn
    Modelo B: torch>=2.3 transformers
    Modelo C: sentence-transformers  (y opcional lightgbm; si no, usa HistGradientBoosting)

Salidas (carpeta --out):
    metricas.csv / metricas.json     -> tabla final para comparar modelos
    predicciones_<modelo>.csv        -> probabilidades sobre el MISMO test (para comparar
                                        después con el Experimento 2: audio/multimodal)
    split_test_ids.csv               -> ids del test, para reutilizar el mismo split
    config.json                      -> configuración usada (reproducibilidad)
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


# ============================================================================
# 0. CONFIGURACIÓN
# ============================================================================
@dataclass
class Config:
    data_path: str | None = None
    out_dir: str = "resultados_exp1"

    # Nombres de columnas
    id_col: str = "llamada_id"
    group_col: str = "cliente_id"
    text_col: str = "texto"
    label_col: str = "etiqueta"
    tabular_cols: list = field(
        default_factory=lambda: ["antiguedad_meses", "factura_media", "n_reclamaciones_12m"]
    )

    # Reproducibilidad y partición
    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.10

    # Idioma del texto ("es" o "en"): solo afecta a la lista de palabras de relleno de TF-IDF
    idioma: str = "es"

    # Candidato A
    tfidf_C_grid: tuple = (0.3, 1.0, 3.0, 10.0)

    # Candidato B
    transformer_name: str = "dccuchile/bert-base-spanish-wwm-cased"  # alternativa ligera: distilbert-base-multilingual-cased
    max_len: int = 128
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5

    # Candidato C
    embedder_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Evaluación
    n_bootstrap: int = 1000      # remuestreos para intervalos de confianza
    latency_samples: int = 200   # nº de llamadas sueltas para medir latencia


# ============================================================================
# 1. UTILIDADES GENERALES
# ============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Elige GPU si existe; así se registra también en los resultados (coste hardware)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ============================================================================
# 2. DATOS: carga, datos de demostración y partición
# ============================================================================
def make_demo_data(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Genera llamadas sintéticas en español SOLO para probar que el código funciona.

    Incluye casos "trampa" para TF-IDF:
      - queja implícita sin palabras clave ("me habéis dejado sin línea otra vez")
      - palabras clave en contexto NO negativo ("¿cómo doy de baja un servicio extra?")
    """
    rng = np.random.default_rng(seed)
    explicit = ["quiero darme de baja", "voy a pedir la portabilidad",
                "esto es una vergüenza, pongo una reclamación", "cancelen mi contrato ya"]
    implicit = ["me habéis dejado sin línea otra vez", "es la tercera vez que llamo y nadie lo soluciona",
                "me voy a otra compañía", "estoy harto de esperar"]
    neutral = ["quería consultar mi factura", "necesito cambiar la dirección de envío",
               "¿cómo activo el roaming?", "gracias, todo claro", "quiero ampliar mis gigas",
               "buenos días, tengo una duda sobre la tarifa"]
    traps = ["¿cómo doy de baja un servicio extra de música?",
             "gracias por resolver mi reclamación, todo correcto",
             "mi hijo pregunta por la portabilidad de su número, pero estamos encantados"]
    filler = ["hola", "buenas tardes", "vale", "perfecto", "pues mire", "a ver"]

    n_clients = max(n // 3, 1)
    rows = []
    for i in range(n):
        angry = rng.random() < 0.30
        parts = [rng.choice(filler)]
        if angry:
            pool = explicit if rng.random() < 0.5 else implicit
            parts.append(rng.choice(pool))
        else:
            parts.append(rng.choice(traps if rng.random() < 0.25 else neutral))
        if rng.random() < 0.5:
            parts.append(rng.choice(neutral))
        y = int(angry)
        if rng.random() < 0.05:           # 5 % de ruido de etiqueta (realismo)
            y = 1 - y
        rows.append({
            "llamada_id": f"c{i:05d}",
            "cliente_id": f"u{rng.integers(0, n_clients):04d}",
            "texto": ", ".join(parts) + ".",
            "etiqueta": y,
            "antiguedad_meses": int(rng.integers(1, 120) * (0.6 if y else 1.0)),
            "factura_media": float(np.round(rng.normal(45 + 10 * y, 12), 2)),
            "n_reclamaciones_12m": int(rng.poisson(0.5 + 1.5 * y)),
        })
    return pd.DataFrame(rows)


def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    for col in (cfg.id_col, cfg.text_col, cfg.label_col):
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria '{col}' en el CSV.")
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str)
    return df


def split_data(df: pd.DataFrame, cfg: Config):
    """Partición train/val/test AGRUPADA por cliente.

    Por qué agrupar: si el mismo cliente aparece en train y en test, el modelo
    "memoriza" a ese cliente y las métricas salen infladas (fuga de información).
    Se usa SIEMPRE el mismo split para los tres candidatos -> comparación justa.
    """
    if cfg.group_col not in df.columns:
        print(f"[aviso] No hay columna '{cfg.group_col}': cada llamada se trata como grupo propio.")
        df = df.assign(**{cfg.group_col: df[cfg.id_col]})

    g1 = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    i_trval, i_test = next(g1.split(df, groups=df[cfg.group_col]))
    trval, test = df.iloc[i_trval], df.iloc[i_test]

    rel_val = cfg.val_size / (1 - cfg.test_size)
    g2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=cfg.seed)
    i_tr, i_val = next(g2.split(trval, groups=trval[cfg.group_col]))
    return (trval.iloc[i_tr].reset_index(drop=True),
            trval.iloc[i_val].reset_index(drop=True),
            test.reset_index(drop=True))


# ============================================================================
# 3. MÉTRICAS DE CALIDAD
# ============================================================================
def expected_calibration_error(y, p, bins: int = 10) -> float:
    """ECE: ¿cuando el modelo dice 80 %, acierta de verdad ~80 % de las veces?
    Importante si luego se usa la probabilidad como "nivel de riesgo"."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def best_threshold(y_val, p_val) -> float:
    """Umbral que maximiza F1 en VALIDACIÓN (nunca en test, para no hacer trampa)."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def compute_metrics(y, p, thr: float) -> dict:
    """Métricas de calidad sobre test, con el umbral fijado previamente en validación."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "umbral": thr,
        "precision": tp / (tp + fp + 1e-12),   # de los que marco en riesgo, ¿cuántos lo son?
        "recall": tp / (tp + fn + 1e-12),      # de los que están en riesgo, ¿cuántos detecto?
        "f1": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),  # métrica principal con clases desbalanceadas
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def _f1_at(y, p, thr):
    return f1_score(y, (p >= thr).astype(int), zero_division=0)


def _prauc(y, p, thr):  # thr se ignora; firma común con _f1_at
    return average_precision_score(y, p)


def bootstrap_ci(y, p, thr, metric_fn, n: int, seed: int):
    """Intervalo de confianza 95 % por bootstrap: ¿cuánto bailaría la métrica con otro test?"""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(metric_fn(y[i], p[i], thr))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_delta(y, p_ref, thr_ref, p_new, thr_new, metric_fn, n: int, seed: int):
    """Diferencia (nuevo - referencia) con remuestreos PAREADOS (mismas llamadas para ambos).
    Si el IC 95 % incluye 0, la mejora no es estadísticamente distinguible del azar."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        deltas.append(metric_fn(y[i], p_new[i], thr_new) - metric_fn(y[i], p_ref[i], thr_ref))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


# ============================================================================
# 4. MÉTRICAS DE COSTE (tiempo de entrenamiento, latencia, tamaño)
# ============================================================================
def measure_latency(infer_one, rows: list, n_warmup: int = 5) -> dict:
    """Latencia de UNA llamada suelta (lo que ocurriría en producción, no en lote).
    Incluye todo el pipeline: tokenizar/vectorizar + modelo + predicción final."""
    for r in rows[:n_warmup]:
        infer_one(r)                      # calentamiento (carga perezosa, cachés, GPU)
    times = []
    for r in rows:
        t0 = time.perf_counter()
        infer_one(r)
        times.append((time.perf_counter() - t0) * 1000)
    return {"lat_ms_p50": float(np.percentile(times, 50)),
            "lat_ms_p95": float(np.percentile(times, 95)),
            "lat_ms_media": float(np.mean(times))}


# ============================================================================
# 5. CANDIDATO A — TF-IDF + REGRESIÓN LOGÍSTICA
# ============================================================================
# Lista corta de palabras de relleno. OJO: NO se incluye "no" ni "sin" porque
# la negación es información valiosa ("no estoy contento").
STOPWORDS = {
    "es": ["el", "la", "los", "las", "de", "del", "y", "a", "en", "un", "una",
           "que", "se", "por", "con", "para", "es", "hola", "vale"],
    "en": ["the", "a", "an", "and", "of", "to", "in", "is", "it", "for", "on",
           "that", "this", "with", "hi", "hello", "ok"],
}


def run_tfidf_logreg(tr, val, te, cfg: Config) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    t0 = time.perf_counter()
    best = None
    # Mini búsqueda del hiperparámetro C (fuerza de regularización) usando VALIDACIÓN
    for C in cfg.tfidf_C_grid:
        pipe = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),       # palabras sueltas y pares ("darme de baja")
                min_df=2,                 # ignora términos que aparecen en <2 llamadas
                sublinear_tf=True,        # suaviza repeticiones de una misma palabra
                strip_accents="unicode",
                lowercase=True,
                stop_words=STOPWORDS.get(cfg.idioma, None),
            ),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
        )
        pipe.fit(tr[cfg.text_col], tr[cfg.label_col])
        p_val = pipe.predict_proba(val[cfg.text_col])[:, 1]
        score = average_precision_score(val[cfg.label_col], p_val)
        if best is None or score > best["score"]:
            best = {"score": score, "C": C, "pipe": pipe, "p_val": p_val}
    train_s = time.perf_counter() - t0

    pipe = best["pipe"]
    p_test = pipe.predict_proba(te[cfg.text_col])[:, 1]

    # Interpretabilidad gratuita: qué palabras empujan hacia "riesgo"
    vec, clf = pipe.steps[0][1], pipe.steps[1][1]
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    top_pos = list(names[order[-10:]][::-1])
    top_neg = list(names[order[:10]])

    return {
        "p_val": best["p_val"], "p_test": p_test, "train_s": train_s,
        "infer_one": lambda r: pipe.predict_proba([r[cfg.text_col]]),
        "n_params": int(clf.coef_.size),
        "size_mb": len(pickle.dumps(pipe)) / 1e6,
        "device": "cpu",
        "extra": {"C_elegido": best["C"], "top_palabras_riesgo": top_pos,
                  "top_palabras_no_riesgo": top_neg},
    }


# ============================================================================
# 6. CANDIDATO B — TRANSFORMER AJUSTADO (FINE-TUNING)
# ============================================================================
def run_transformer_finetune(tr, val, te, cfg: Config) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_seed(cfg.seed)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(cfg.transformer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.transformer_name, num_labels=2).to(device)

    def encode(texts):
        return tok(list(texts), truncation=True, max_length=cfg.max_len,
                   padding=True, return_tensors="pt")

    @torch.no_grad()
    def predict_proba(texts, bs: int = 64):
        model.eval()
        out = []
        for i in range(0, len(texts), bs):
            enc = {k: v.to(device) for k, v in encode(texts[i:i + bs]).items()}
            logits = model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    # Pérdida ponderada para compensar el desbalanceo de clases
    y_tr = tr[cfg.label_col].values
    counts = np.bincount(y_tr, minlength=2)
    weights = torch.tensor(len(y_tr) / (2 * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    steps = cfg.epochs * int(np.ceil(len(tr) / cfg.batch_size))
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
    use_amp = device == "cuda"                      # precisión mixta solo en GPU NVIDIA
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    texts_tr = tr[cfg.text_col].tolist()
    idx_all = np.arange(len(tr))
    best = {"score": -1.0, "state": None, "epoch": -1}

    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        np.random.shuffle(idx_all)
        for i in range(0, len(idx_all), cfg.batch_size):
            b = idx_all[i:i + cfg.batch_size]
            enc = {k: v.to(device) for k, v in encode([texts_tr[j] for j in b]).items()}
            yb = torch.tensor(y_tr[b], dtype=torch.long, device=device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**enc).logits.float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()

        # Evaluar en validación al final de cada época y guardar la mejor
        p_val = predict_proba(val[cfg.text_col].tolist())
        score = average_precision_score(val[cfg.label_col], p_val)
        print(f"   [B] época {epoch + 1}/{cfg.epochs}  PR-AUC val = {score:.4f}")
        if score > best["score"]:
            best = {"score": score, "epoch": epoch + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    train_s = time.perf_counter() - t0

    model.load_state_dict(best["state"])             # recuperar la mejor época
    p_val = predict_proba(val[cfg.text_col].tolist())
    p_test = predict_proba(te[cfg.text_col].tolist())
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "p_val": p_val, "p_test": p_test, "train_s": train_s,
        "infer_one": lambda r: predict_proba([r[cfg.text_col]], bs=1),
        "n_params": int(n_params),
        "size_mb": n_params * 4 / 1e6,               # fp32
        "device": device,
        "extra": {"mejor_epoca": best["epoch"], "modelo_base": cfg.transformer_name},
    }


# ============================================================================
# 7. CANDIDATO C — EMBEDDINGS CONGELADOS + GRADIENT BOOSTING
# ============================================================================
def run_embeddings_gb(tr, val, te, cfg: Config, use_tabular: bool = False) -> dict:
    from sentence_transformers import SentenceTransformer

    set_seed(cfg.seed)
    device = get_device()
    embedder = SentenceTransformer(cfg.embedder_name, device=device)  # congelado: no se reentrena
    tab_cols = [c for c in cfg.tabular_cols if c in tr.columns] if use_tabular else []

    def embed(texts):
        return embedder.encode(list(texts), batch_size=64, normalize_embeddings=True,
                               show_progress_bar=False)

    def build_X(emb, frame):
        """Concatena embedding (texto) + columnas de la ficha del cliente (si procede)."""
        if not tab_cols:
            return emb
        return np.hstack([emb, frame[tab_cols].astype(float).values])

    t0 = time.perf_counter()
    X_tr = build_X(embed(tr[cfg.text_col]), tr)
    X_val = build_X(embed(val[cfg.text_col]), val)
    X_te = build_X(embed(te[cfg.text_col]), te)
    t_embed = time.perf_counter() - t0               # coste de "traducir" a números

    t1 = time.perf_counter()
    try:
        import lightgbm as lgb
        gb = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5,
            subsample=0.8, subsample_freq=1, class_weight="balanced",
            random_state=cfg.seed, verbose=-1)
        gb.fit(X_tr, tr[cfg.label_col], eval_set=[(X_val, val[cfg.label_col])],
               eval_metric="average_precision",
               callbacks=[lgb.early_stopping(30, verbose=False)])
        gb_name = "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gb = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, early_stopping=True,
            class_weight="balanced", random_state=cfg.seed)
        gb.fit(X_tr, tr[cfg.label_col])
        gb_name = "HistGradientBoosting (sklearn)"
    t_gb = time.perf_counter() - t1

    p_val = gb.predict_proba(X_val)[:, 1]
    p_test = gb.predict_proba(X_te)[:, 1]

    def infer_one(r):
        """Una llamada suelta: embedding + (ficha) + boosting."""
        e = embedder.encode([r[cfg.text_col]], normalize_embeddings=True,
                            show_progress_bar=False)
        if tab_cols:
            e = np.hstack([e, np.array([[float(r[c]) for c in tab_cols]])])
        return gb.predict_proba(e)

    n_emb_params = sum(p.numel() for p in embedder.parameters())
    return {
        "p_val": p_val, "p_test": p_test, "train_s": t_embed + t_gb,
        "infer_one": infer_one,
        "n_params": int(n_emb_params),               # solo el embedder; el GB es pequeño
        "size_mb": n_emb_params * 4 / 1e6 + len(pickle.dumps(gb)) / 1e6,
        "device": device,
        "extra": {"boosting": gb_name, "tiempo_embeddings_s": round(t_embed, 2),
                  "tiempo_boosting_s": round(t_gb, 2), "cols_ficha": tab_cols,
                  "embedder": cfg.embedder_name},
    }


# ============================================================================
# 8. ORQUESTADOR: entrena, evalúa y compara
# ============================================================================
def evaluate_model(name: str, res: dict, tr, val, te, cfg: Config, outdir: Path) -> dict:
    y_val = val[cfg.label_col].values
    y_te = te[cfg.label_col].values

    thr = best_threshold(y_val, res["p_val"])        # umbral fijado en validación
    m = compute_metrics(y_te, res["p_test"], thr)

    # Intervalos de confianza de las dos métricas clave
    m["f1_ic95"] = bootstrap_ci(y_te, res["p_test"], thr, _f1_at, cfg.n_bootstrap, cfg.seed)
    m["pr_auc_ic95"] = bootstrap_ci(y_te, res["p_test"], thr, _prauc, cfg.n_bootstrap, cfg.seed)

    # Costes
    rows = te.head(cfg.latency_samples).to_dict("records")
    m.update(measure_latency(res["infer_one"], rows))
    m.update({"train_s": res["train_s"], "n_params": res["n_params"],
              "size_mb": res["size_mb"], "device": res["device"]})
    m["modelo"] = name
    m["extra"] = res["extra"]

    # Guardar predicciones para comparar luego con el Experimento 2 (mismas llamadas)
    pd.DataFrame({
        cfg.id_col: te[cfg.id_col], "y_true": y_te,
        "proba": res["p_test"], "pred": (res["p_test"] >= thr).astype(int),
    }).to_csv(outdir / f"predicciones_{name}.csv", index=False)
    return m


def compare_against_baseline(results: dict, y_te, cfg: Config) -> pd.DataFrame:
    """Compara cada modelo contra el candidato A: ¿cuánto mejora y a qué precio?"""
    ref = "A"
    rows = []
    for name, r in results.items():
        if name == ref:
            continue
        d_f1 = paired_bootstrap_delta(y_te, results[ref]["p_test"], results[ref]["umbral"],
                                      r["p_test"], r["umbral"], _f1_at, cfg.n_bootstrap, cfg.seed)
        d_pr = paired_bootstrap_delta(y_te, results[ref]["p_test"], results[ref]["umbral"],
                                      r["p_test"], r["umbral"], _prauc, cfg.n_bootstrap, cfg.seed)
        significativa = d_f1[1] > 0                   # IC 95 % de ΔF1 completamente por encima de 0
        extra_ms = r["lat_ms_p50"] - results[ref]["lat_ms_p50"]
        extra_train = r["train_s"] - results[ref]["train_s"]
        rows.append({
            "modelo": name,
            "delta_f1": d_f1[0], "delta_f1_ic95": (d_f1[1], d_f1[2]),
            "delta_pr_auc": d_pr[0], "delta_pr_auc_ic95": (d_pr[1], d_pr[2]),
            "mejora_significativa": significativa,
            "latencia_extra_ms": extra_ms,
            "entrenamiento_extra_s": extra_train,
            "ms_extra_por_punto_f1": (extra_ms / (d_f1[0] * 100)) if d_f1[0] > 0 else np.nan,
            "veredicto": ("Merece la pena estudiarlo" if significativa
                          else "No supera con claridad a la línea base: quedarse con A"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimento 1: línea base textual")
    ap.add_argument("--data", type=str, help="CSV con llamadas")
    ap.add_argument("--demo", action="store_true", help="usar datos sintéticos de prueba")
    ap.add_argument("--models", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--out", type=str, default="resultados_exp1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--transformer", type=str, default=None)
    ap.add_argument("--embedder", type=str, default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--lat-n", type=int, default=200)
    ap.add_argument("--idioma", choices=["es", "en"], default="es",
                    help="idioma del texto (palabras de relleno de TF-IDF)")
    args = ap.parse_args()

    cfg = Config(data_path=args.data, out_dir=args.out, seed=args.seed, epochs=args.epochs,
                 n_bootstrap=args.bootstrap, latency_samples=args.lat_n, idioma=args.idioma)
    if args.transformer:
        cfg.transformer_name = args.transformer
    if args.embedder:
        cfg.embedder_name = args.embedder

    outdir = Path(cfg.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    # ---- Datos ----
    if args.demo:
        df = make_demo_data(seed=cfg.seed)
        print("[demo] Datos sintéticos (solo para probar el código; las métricas NO son reales).")
    elif cfg.data_path:
        df = load_data(cfg)
    else:
        ap.error("Indica --data <csv> o --demo")

    tr, val, te = split_data(df, cfg)
    print(f"Split -> train={len(tr)}  val={len(val)}  test={len(te)} | "
          f"% positivos train={tr[cfg.label_col].mean():.1%} test={te[cfg.label_col].mean():.1%}")
    te[[cfg.id_col]].to_csv(outdir / "split_test_ids.csv", index=False)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    # ---- Candidatos ----
    runners = {}
    if "A" in args.models:
        runners["A"] = lambda: run_tfidf_logreg(tr, val, te, cfg)
    if "B" in args.models:
        runners["B"] = lambda: run_transformer_finetune(tr, val, te, cfg)
    if "C" in args.models:
        runners["C"] = lambda: run_embeddings_gb(tr, val, te, cfg, use_tabular=False)
        if any(c in tr.columns for c in cfg.tabular_cols):
            # Variante C + ficha: muestra cuánto aporta mezclar texto y datos tabulares
            runners["C_ficha"] = lambda: run_embeddings_gb(tr, val, te, cfg, use_tabular=True)

    results = {}
    for name, runner in runners.items():
        print(f"\n>>> Entrenando candidato {name} ...")
        try:
            res = runner()
        except ImportError as e:
            print(f"   [omitido] Falta una dependencia para {name}: {e}")
            continue
        results[name] = evaluate_model(name, res, tr, val, te, cfg, outdir)
        r = results[name]
        # p_test se conserva en memoria para la comparación pareada
        r["p_test"] = res["p_test"]
        print(f"   F1={r['f1']:.3f}  PR-AUC={r['pr_auc']:.3f}  recall={r['recall']:.3f}  "
              f"lat p50={r['lat_ms_p50']:.1f} ms  train={r['train_s']:.1f} s")

    if not results:
        print("No se ejecutó ningún modelo.")
        return

    # ---- Tabla final de resultados ----
    cols = ["modelo", "f1", "f1_ic95", "pr_auc", "pr_auc_ic95", "roc_auc", "precision", "recall",
            "f1_macro", "brier", "ece", "umbral", "lat_ms_p50", "lat_ms_p95", "train_s",
            "n_params", "size_mb", "device"]
    table = pd.DataFrame([{k: v for k, v in r.items()} for r in results.values()])[cols]
    for c in ("f1_ic95", "pr_auc_ic95"):               # (lo, hi) -> "[lo, hi]" legible
        table[c] = table[c].apply(lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]")
    table.to_csv(outdir / "metricas.csv", index=False)
    serializable = {k: {kk: vv for kk, vv in v.items() if kk != "p_test"} for k, v in results.items()}
    (outdir / "metricas.json").write_text(json.dumps(serializable, indent=2, default=str))

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n" + "=" * 90 + "\nRESULTADOS EN TEST\n" + "=" * 90)
    print(table.round(3).to_string(index=False))

    if "A" in results and len(results) > 1:
        cmp_df = compare_against_baseline(results, te[cfg.label_col].values, cfg)
        cmp_df.to_csv(outdir / "comparacion_vs_A.csv", index=False)
        print("\n" + "=" * 90 + "\nCOMPARACIÓN FRENTE A LA LÍNEA BASE (A)\n" + "=" * 90)
        print(cmp_df.round(3).to_string(index=False))

    if "A" in results:
        ex = results["A"]["extra"]
        print("\nPalabras que más empujan a 'riesgo' (A):", ", ".join(ex["top_palabras_riesgo"]))

    print(f"\nResultados guardados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
