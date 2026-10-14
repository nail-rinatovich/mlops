"""Обучает классификатор Iris и экспортирует его в ONNX.

Артефакты (в каталоге --out):
  model.onnx     — модель, вход "input" float32[N,4], выходы "label" int64[N] и "probabilities" float32[N,3]
  metadata.json  — имена фичей/классов, допустимые диапазоны, метрики
  samples.json   — эталонные входы и ожидаемые выходы для проверки паритета в Go
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
from skl2onnx import to_onnx
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/out")
    parser.add_argument("--version", default="1")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
    }

    # zipmap=False: вероятности как обычный тензор [N, n_classes], а не список словарей.
    onx = to_onnx(
        clf,
        X_train[:1],
        options={id(clf.steps[-1][1]): {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3},
        final_types=None,
    )
    # Переименуем входы/выходы в стабильные имена — на них опирается Go-сервис.
    rename = {onx.graph.input[0].name: "input",
              onx.graph.output[0].name: "label",
              onx.graph.output[1].name: "probabilities"}
    for node in onx.graph.node:
        node.input[:] = [rename.get(n, n) for n in node.input]
        node.output[:] = [rename.get(n, n) for n in node.output]
    for v in list(onx.graph.input) + list(onx.graph.output):
        v.name = rename.get(v.name, v.name)

    model_bytes = onx.SerializeToString()
    (out / "model.onnx").write_bytes(model_bytes)

    # Допустимые диапазоны: [0, 2*max] по обучающей выборке — грубая защита от мусора.
    maxs = X_train.max(axis=0)
    metadata = {
        "name": "iris",
        "version": args.version,
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "features": [
            {"name": n.replace(" (cm)", "").replace(" ", "_"), "min": 0.0, "max": round(float(m) * 2, 2)}
            for n, m in zip(iris.feature_names, maxs)
        ],
        "classes": [str(c) for c in iris.target_names],
        "metrics": metrics,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    sample_x = X_test[:10]
    samples = {
        "inputs": sample_x.tolist(),
        "labels": clf.predict(sample_x).tolist(),
        "probabilities": clf.predict_proba(sample_x).astype(np.float32).tolist(),
    }
    (out / "samples.json").write_text(json.dumps(samples, indent=2))
    print(json.dumps({"out": str(out), **metrics}))


if __name__ == "__main__":
    main()
