#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compara tres modelos para clasificar emociones en textos del MEACorpus.

Por defecto usa spanish-meacorpus-2023-dataset.csv, situado junto a este
archivo. El CSV debe tener una columna de texto y otra con la etiqueta.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_DATA = Path(__file__).with_name("spanish-meacorpus-2023-dataset.csv")


def load_data(path: Path, text_col: str, label_col: str):
    """Carga el CSV y usa su división train/test si está disponible."""
    df = pd.read_csv(path, encoding="utf-8")
    for column in (text_col, label_col):
        if column not in df.columns:
            raise ValueError(f"No se encontró la columna '{column}' en {path}.")

    df = df.dropna(subset=[label_col]).copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[label_col] = df[label_col].astype(str)

    if "split" in df.columns and {"train", "test"}.issubset(
        set(df["split"].astype(str).str.lower())
    ):
        train_all = df[df["split"].str.lower() == "train"]
        test = df[df["split"].str.lower() == "test"]
        train, val = train_test_split(
            train_all,
            test_size=0.15,
            random_state=42,
            stratify=train_all[label_col],
        )
    else:
        train_val, test = train_test_split(
            df, test_size=0.20, random_state=42, stratify=df[label_col]
        )
        train, val = train_test_split(
            train_val,
            test_size=0.15,
            random_state=42,
            stratify=train_val[label_col],
        )

    encoder = LabelEncoder()
    encoder.fit(df[label_col])
    return train, val, test, encoder


def score_model(name, y_true, y_pred, classes, seconds):
    return {
        "modelo": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "tiempo_entrenamiento_segundos": seconds,
    }


def run_tfidf(train, val, test, text_col, label_col, encoder):
    """Modelo A: palabras + regresión logística."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    best_model = None
    best_score = -1
    start = time.perf_counter()
    for strength in (0.1, 1, 10):
        model = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, strip_accents="unicode"
            ),
            LogisticRegression(
                C=strength, class_weight="balanced", max_iter=1000
            ),
        )
        model.fit(train[text_col], train[label_col])
        val_pred = model.predict(val[text_col])
        score = f1_score(val[label_col], val_pred, average="macro")
        if score > best_score:
            best_model, best_score = model, score

    seconds = time.perf_counter() - start
    test_pred = best_model.predict(test[text_col])
    return score_model(
        "A - TF-IDF", test[label_col], test_pred, encoder.classes_, seconds
    ), test_pred


def run_transformer(train, val, test, text_col, label_col, encoder, epochs):
    """Modelo B: transformer ligero congelado y una capa de clasificación."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert/distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(encoder.classes_)
    ).to(device)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    label_ids = {label: i for i, label in enumerate(encoder.classes_)}

    def predict(texts):
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = tokenizer(
                    list(texts[start : start + 32]),
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                predictions.extend(model(**batch).logits.argmax(dim=1).cpu().tolist())
        return np.asarray(predictions)

    y_train = np.asarray([label_ids[label] for label in train[label_col]])
    counts = np.bincount(y_train, minlength=len(encoder.classes_))
    weights = torch.tensor(
        len(y_train) / (len(counts) * np.maximum(counts, 1)),
        dtype=torch.float,
        device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=5e-4,
    )
    best_state = None
    best_score = -1
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        model.base_model.eval()
        order = np.random.permutation(len(train))
        for offset in range(0, len(order), 8):
            indices = order[offset : offset + 8]
            batch = tokenizer(
                train.iloc[indices][text_col].tolist(),
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = torch.tensor(y_train[indices], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            optimizer.step()

        val_pred = predict(val[text_col].tolist())
        val_true = np.asarray([label_ids[label] for label in val[label_col]])
        score = f1_score(val_true, val_pred, average="macro")
        print(f"  Época {epoch + 1}/{epochs}: F1 macro validación = {score:.3f}")
        if score > best_score:
            best_score = score
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

    seconds = time.perf_counter() - start
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_state:
                parameter.copy_(best_state[name].to(device))
    test_pred = predict(test[text_col].tolist())
    test_pred = encoder.inverse_transform(test_pred)
    return score_model(
        "B - Transformer", test[label_col], test_pred, encoder.classes_, seconds
    ), test_pred


def run_embeddings(train, val, test, text_col, label_col, encoder):
    """Modelo C: embeddings de texto + Gradient Boosting."""
    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier

    start = time.perf_counter()
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    x_train = embedder.encode(train[text_col].tolist(), show_progress_bar=False)
    x_test = embedder.encode(test[text_col].tolist(), show_progress_bar=False)
    model = HistGradientBoostingClassifier(
        max_iter=200, class_weight="balanced", random_state=42
    )
    model.fit(x_train, encoder.transform(train[label_col]))
    seconds = time.perf_counter() - start
    test_pred = encoder.inverse_transform(model.predict(x_test))
    return score_model(
        "C - Embeddings + Boosting",
        test[label_col],
        test_pred,
        encoder.classes_,
        seconds,
    ), test_pred


def main():
    parser = argparse.ArgumentParser(
        description="Compara tres modelos de clasificación de texto."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV de datos")
    parser.add_argument("--text-col", default="transcription", help="Columna de texto")
    parser.add_argument("--label-col", default="label", help="Columna de etiqueta")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="Modelos que se ejecutarán",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Épocas del transformer")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="entrena con una muestra pequeña para comprobar que funciona",
    )
    parser.add_argument("--out", type=Path, default=Path("resultados_exp1"))
    args = parser.parse_args()

    train, val, test, encoder = load_data(
        args.data, args.text_col, args.label_col
    )
    if args.quick:
        per_class = max(1, 500 // len(encoder.classes_))
        train = pd.concat(
            [
                group.sample(
                    n=min(len(group), per_class),
                    random_state=42,
                )
                for _, group in train.groupby(args.label_col)
            ]
        ).sample(frac=1, random_state=42).reset_index(drop=True)
        print("Modo de prueba rápida: se usa una muestra pequeña de entrenamiento.")
    print(
        f"Datos: {len(train)} entrenamiento, {len(val)} validación, "
        f"{len(test)} prueba; {len(encoder.classes_)} emociones."
    )

    runners = {
        "A": lambda: run_tfidf(
            train, val, test, args.text_col, args.label_col, encoder
        ),
        "B": lambda: run_transformer(
            train, val, test, args.text_col, args.label_col, encoder, args.epochs
        ),
        "C": lambda: run_embeddings(
            train, val, test, args.text_col, args.label_col, encoder
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.models:
        print(f"Entrenando modelo {name}...")
        try:
            metrics, predictions = runners[name]()
        except ImportError as error:
            print(f"Se omite el modelo {name}: falta una dependencia ({error}).")
            continue
        results.append(metrics)
        pd.DataFrame(
            {
                "texto": test[args.text_col].to_numpy(),
                "etiqueta_real": test[args.label_col].to_numpy(),
                "etiqueta_predicha": predictions,
            }
        ).to_csv(args.out / f"predicciones_{name}.csv", index=False)
        print(f"  F1 macro: {metrics['f1_macro']:.3f}")

    if not results:
        raise RuntimeError("No se pudo ejecutar ningún modelo.")

    table = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    table.to_csv(args.out / "comparacion.csv", index=False)
    print("\nComparación (mejor F1 macro primero):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nResultados guardados en: {args.out.resolve()}")


if __name__ == "__main__":
    main()
