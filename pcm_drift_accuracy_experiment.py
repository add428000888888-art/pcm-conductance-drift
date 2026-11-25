"""Reproduce the Digits simulation using the versioned Cai et al. source data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import urllib.request
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

LEVELS = np.array([5, 10, 15, 20, 25, 35, 40, 45], dtype=np.float32)
DATA_URL = "https://zenodo.org/records/19162997/files/Data.zip?download=1"
DATA_SHA256 = "85b818aed32dc9df992c983a77a38d749e9d1dff6b5e0b2e4d96c2ec00bbe233"
SPLIT_SEED = 42
TRAINING_SEEDS = range(42, 52)
SEVERITIES = np.linspace(0, 1, 11)
TRIALS = 30


def load_ratios(payload):
    """Parse corresponding rows within adjacent, header-validated columns."""
    pools = {}
    rows = []
    with ZipFile(io.BytesIO(payload)) as archive:
        for number, level in enumerate(LEVELS[::-1], 1):
            member = f"Data/Fig2/D/Fig2D{number}.xlsx"
            with archive.open(member) as workbook:
                frame = pd.read_excel(workbook, engine="openpyxl")
            columns = [re.sub(r"\.\d+$", "", str(name)) for name in frame]
            expected = ["Origin Conductance(logμS)", "Drift Conductance(logμS)"]
            if not columns or len(columns) % 2 or columns != expected * (len(columns) // 2):
                raise ValueError(f"Unexpected column layout in {member}")
            values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            origin_log, drift_log = values[:, 0::2], values[:, 1::2]
            valid = np.isfinite(origin_log) & np.isfinite(drift_log)
            origin = 10.0 ** origin_log[valid]
            drift = 10.0 ** drift_log[valid]
            # Keep the source script's precision and row-major sampling order.
            ratios = drift / origin
            if ratios.size < 2 or not np.isfinite(ratios).all() or (ratios <= 0).any():
                raise ValueError(f"Invalid conductance ratio pool in {member}")
            if LEVELS[np.abs(LEVELS - origin.mean()).argmin()] != level:
                raise ValueError(f"Nominal conductance does not match {member}")
            pools[float(level)] = ratios.astype(np.float32)
            rows.append({
                "level_uS": float(level),
                "workbook": member,
                "column_pairs": len(columns) // 2,
                "rows": len(frame),
                "pairs": ratios.size,
                "excluded_pairs": int(valid.size - valid.sum()),
                "origin_mean_uS": origin.mean(),
                "drift_mean_uS": drift.mean(),
                "ratio_mean": ratios.mean(),
                "ratio_sd": ratios.std(ddof=1),
                "mean_decrease_percent": 100 * (1 - ratios.mean()),
                "increase_percent": 100 * (drift_log[valid] > origin_log[valid]).mean(),
                "ratio_p05": np.quantile(ratios, 0.05),
                "ratio_p95": np.quantile(ratios, 0.95),
            })
    return pools, pd.DataFrame(rows).sort_values("level_uS")


def cnn_patches(x):
    windows = sliding_window_view(x.reshape(-1, 8, 8), (3, 3), axis=(1, 2))
    return np.ascontiguousarray(windows.reshape(-1, 36, 9))


def forward_cnn(patches, weights, biases, training=False):
    kernels, hidden_weights, output_weights = weights
    conv = patches @ kernels + biases[0]
    # Arrange each 2 × 2 pooling window in the last dimension.
    windows = conv.reshape(-1, 3, 2, 3, 2, 8).transpose(0, 1, 3, 5, 2, 4)
    windows = windows.reshape(-1, 3, 3, 8, 4)
    pooled = np.maximum(windows.max(axis=-1), 0)
    features = pooled.reshape(-1, 72)
    hidden = np.maximum(features @ hidden_weights + biases[1], 0)
    logits = hidden @ output_weights + biases[2]
    if training:
        return logits, (windows, pooled, features, hidden)
    return logits


def cnn_gradients(patches, labels, weights, biases):
    logits, (windows, pooled, features, hidden) = forward_cnn(
        patches, weights, biases, training=True
    )
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    delta_output = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    delta_output[np.arange(len(labels)), labels] -= 1
    delta_output /= len(labels)
    delta_hidden = (delta_output @ weights[2].T) * (hidden > 0)
    delta_pooled = (delta_hidden @ weights[1].T).reshape(pooled.shape)
    delta_windows = np.zeros_like(windows)
    np.put_along_axis(
        delta_windows, windows.argmax(axis=-1, keepdims=True),
        (delta_pooled * (pooled > 0))[..., None], axis=-1
    )
    delta_conv = delta_windows.reshape(-1, 3, 3, 8, 2, 2).transpose(0, 1, 4, 2, 5, 3)
    delta_conv = delta_conv.reshape(-1, 8)
    return [
        patches.reshape(-1, 9).T @ delta_conv,
        features.T @ delta_hidden,
        hidden.T @ delta_output,
        delta_conv.sum(axis=0),
        delta_hidden.sum(axis=0),
        delta_output.sum(axis=0),
    ]


def train_cnn(x, y, train, seed):
    train, validation = train_test_split(
        train, test_size=0.10, stratify=y[train], random_state=SPLIT_SEED
    )
    rng = np.random.default_rng(seed)
    patches = cnn_patches(x)
    shapes = ((9, 8), (72, 32), (32, 10))
    weights = [
        rng.standard_normal(shape, dtype=np.float32) * np.sqrt(gain / shape[0]).astype(np.float32)
        for shape, gain in zip(shapes, (2, 2, 1), strict=True)
    ]
    biases = [np.zeros(shape[1], dtype=np.float32) for shape in shapes]
    parameters = weights + biases
    first = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    best_score, best_epoch, step = -1, 0, 0
    for epoch in range(1, 81):
        order = rng.permutation(train)
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            gradients = cnn_gradients(patches[batch], y[batch], weights, biases)
            step += 1
            for p, g, m, v in zip(parameters, gradients, first, second, strict=True):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g ** 2
                p -= 0.002 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        score = np.count_nonzero(
            forward_cnn(patches[validation], weights, biases).argmax(axis=1) == y[validation]
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best = [p.copy() for p in parameters]
        if epoch - best_epoch >= 15:
            break
    return best[:3], best[3:], {"best_epoch": best_epoch, "epochs_run": epoch}


def forward_mlp(x, weights, biases):
    hidden = np.maximum(x @ weights[0] + biases[0], 0)
    return hidden @ weights[1] + biases[1]


def map_weights(weight):
    peak = float(np.max(np.abs(weight)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Expected a finite weight tensor with nonzero magnitude")
    span = float(LEVELS[-1] - LEVELS[0])
    scale = span / peak
    plus = LEVELS[0] + scale * np.maximum(weight, 0)
    minus = LEVELS[0] + scale * np.maximum(-weight, 0)
    # argmin resolves any exact tie in favor of the lower conductance.
    plus_index = np.abs(plus[..., None] - LEVELS).argmin(axis=-1)
    minus_index = np.abs(minus[..., None] - LEVELS).argmin(axis=-1)
    return plus_index, minus_index, 1 / scale


def sample_ratios(indices, pools, rng):
    sampled = np.empty(indices.shape, dtype=np.float32)
    for index, level in enumerate(LEVELS):
        mask = indices == index
        sampled[mask] = rng.choice(pools[float(level)], size=int(mask.sum()), replace=True)
    return sampled


def effective_weights(mappings, ratios, severity):
    return [
        (LEVELS[plus] * rp ** severity - LEVELS[minus] * rm ** severity) * inverse
        for (plus, minus, inverse), (rp, rm) in zip(mappings, ratios, strict=True)
    ]


def evaluate(predict, weights, truth, pools, rng_for_trial):
    mappings = [map_weights(w) for w in weights]
    means = np.array([pools[float(level)].mean(dtype=np.float64) for level in LEVELS], dtype=np.float32)
    mean_ratios = [(means[plus], means[minus]) for plus, minus, _ in mappings]
    floating = predict(weights)
    baseline = predict(effective_weights(mappings, mean_ratios, 0.0))
    control = predict(effective_weights(mappings, mean_ratios, 1.0))

    def record(condition, trial, severity, predictions):
        return {
            "condition": condition, "trial": trial, "severity": severity,
            "correct": int((predictions == truth).sum()),
            "test_count": len(truth),
            "flips_vs_quantized": int((predictions != baseline).sum()),
            "new_errors_vs_quantized": int(((baseline == truth) & (predictions != truth)).sum()),
            "corrected_vs_quantized": int(((baseline != truth) & (predictions == truth)).sum()),
        }

    yield record("floating", -1, np.nan, floating)
    yield record("quantized", -1, 0.0, baseline)
    yield record("state_mean", -1, 1.0, control)
    for trial in range(TRIALS):
        rng = rng_for_trial(trial)
        ratios = [(sample_ratios(p, pools, rng), sample_ratios(m, pools, rng)) for p, m, _ in mappings]
        for severity in SEVERITIES[1:]:
            predictions = predict(effective_weights(mappings, ratios, float(severity)))
            yield record("empirical", trial, severity, predictions)


def summarize(scores):
    rows = []
    for (model, seed), group in scores.groupby(["model", "training_seed"], sort=False):
        count = group.test_count.iloc[0]
        quantized = group[group.condition == "quantized"].correct.iloc[0]
        floating = group[group.condition == "floating"].correct.iloc[0]
        mean_control = group[group.condition == "state_mean"].correct.iloc[0]
        endpoint = group[(group.condition == "empirical") & (group.severity == 1)]
        rows.append({
            "model": model, "training_seed": seed,
            "float_percent": 100 * floating / count,
            "quantized_percent": 100 * quantized / count,
            "endpoint_mean_percent": 100 * endpoint.correct.mean() / count,
            "mapping_loss_pp": 100 * (floating - quantized) / count,
            "endpoint_loss_pp": 100 * (quantized - endpoint.correct.mean()) / count,
            "endpoint_trial_sd_pp": 100 * endpoint.correct.std(ddof=1) / count,
            "state_mean_loss_pp": 100 * (quantized - mean_control) / count,
            "mean_prediction_flips": endpoint.flips_vs_quantized.mean(),
            "mean_new_errors": endpoint.new_errors_vs_quantized.mean(),
            "mean_corrected": endpoint.corrected_vs_quantized.mean(),
        })
    return pd.DataFrame(rows)


def run(pools, out):
    digits = load_digits()
    x = (digits.data / 16.0).astype(np.float32)
    y = digits.target
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=SPLIT_SEED, stratify=y
    )
    np.savez_compressed(out / "split.npz", train=train, test=test)
    patches = cnn_patches(x[test])
    records, legacy, training = [], [], []
    for seed in TRAINING_SEEDS:
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", solver="adam",
            alpha=1e-4, max_iter=500, early_stopping=True,
            n_iter_no_change=25, random_state=seed, batch_size=200,
            validation_fraction=0.1, learning_rate_init=0.001,
        ).fit(x[train], y[train])
        mlp_biases = mlp.intercepts_
        cnn_weights, cnn_biases, cnn_history = train_cnn(x, y, train, seed)
        for model_id, (name, weights, biases, forward, inputs, history) in enumerate([
            ("MLP", mlp.coefs_, mlp_biases, forward_mlp, x[test], {"epochs_run": mlp.n_iter_}),
            ("CNN", cnn_weights, cnn_biases, forward_cnn, patches, cnn_history),
        ]):
            def predict(w):
                return forward(inputs, w, biases).argmax(axis=1)

            if name == "MLP":
                np.testing.assert_array_equal(predict(weights), mlp.predict(inputs))
            np.savez_compressed(
                out / f"{name.lower()}_{seed}.npz",
                **{f"weight_{i}": w for i, w in enumerate(weights)},
                **{f"bias_{i}": b for i, b in enumerate(biases)},
            )
            training.append({"model": name, "training_seed": seed, **history})
            rows = evaluate(
                predict, weights, y[test], pools,
                lambda trial: np.random.default_rng(np.random.SeedSequence([42000, seed, model_id, trial])),
            )
            records.extend({"model": name, "training_seed": seed, **row} for row in rows)
            if seed == SPLIT_SEED and name == "MLP":
                for offset in (10000, 20000):
                    rows = evaluate(
                        predict, weights, y[test], pools,
                        lambda trial: np.random.default_rng(SPLIT_SEED + offset + trial),
                    )
                    legacy.extend({"model": name, "training_seed": seed, "device_seed_offset": offset, **row} for row in rows)
        print(f"Finished training seed {seed}", flush=True)
    scores = pd.DataFrame(records)
    scores.to_csv(out / "trial_scores.csv", index=False)
    pd.DataFrame(training).to_csv(out / "training.csv", index=False)
    pd.DataFrame(legacy).to_csv(out / "legacy_mlp_trial_scores.csv", index=False)
    summarize(scores).to_csv(out / "seed_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local copy of the version 1 Zenodo Data.zip")
    parser.add_argument("--out", type=Path, default=Path("run"), help="New output directory")
    args = parser.parse_args()
    if args.data is None:
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            payload = response.read()
    else:
        payload = args.data.read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
        raise ValueError("Input SHA-256 does not match Zenodo record 19162997 version 1 Data.zip")
    pools, summary = load_ratios(payload)
    args.out.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.out / "device_summary.csv", index=False)
    with threadpool_limits(limits=1):
        run(pools, args.out)
    manifest = {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "threadpoolctl"]},
        "data_doi": "10.5281/zenodo.19162997", "data_sha256": DATA_SHA256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outer_split_seed": SPLIT_SEED, "training_seeds": list(TRAINING_SEEDS),
        "device_trials_per_model": TRIALS,
        "severity": SEVERITIES.tolist(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
