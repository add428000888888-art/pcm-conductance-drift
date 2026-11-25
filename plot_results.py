"""Regenerate the manuscript figures from the archived CSVs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

MODELS = {"MLP": "#215C8D", "CNN": "#A24326"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "pdf.fonttype": 42, "axes.axisbelow": True,
})


def label(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)


def save(fig, directory, name):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", facecolor="white")
    plt.close(fig)


def plot_workflow(directory):
    fig, ax = plt.subplots(figsize=(7.25, 1.20))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    steps = [
        ("Device selection", "Memory hierarchy\nto PCM"),
        ("Crossbar model", "Differential weight\nrepresentation"),
        ("Exploratory sweeps", "Quantization, noise\nand assumed drift"),
        ("Endpoint simulation", "State-specific ratios\nand digit inference"),
    ]
    for index, (title, detail) in enumerate(steps):
        left = 0.015 + 0.25 * index
        ax.add_patch(FancyBboxPatch(
            (left, 0.14), 0.22, 0.72, boxstyle="round,pad=0.005",
            facecolor="#f1f5f8", edgecolor="#a4b8c7", linewidth=0.8,
        ))
        ax.text(left + 0.11, 0.64, title, ha="center", va="center", fontweight="bold")
        ax.text(left + 0.11, 0.36, detail, ha="center", va="center", color="#33495a")
        if index < len(steps) - 1:
            ax.annotate("", (left + 0.247, 0.50), (left + 0.225, 0.50),
                        arrowprops={"arrowstyle": "->", "color": "#526d80", "lw": 0.8})
    save(fig, directory, "fig1_workflow")


def plot_crossbar(directory):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.25, 2.55))
    fig.subplots_adjust(left=0.025, right=0.98, top=0.83, bottom=0.04, wspace=0.16)
    for ax, title in ((a, "a   Ideal conductance array"), (b, "b   Differential pair")):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    row_color, col_color = "#215C8D", "#A24326"
    x_nodes = (0.34, 0.58, 0.82)
    y_nodes = (0.77, 0.55, 0.33)
    for index, y in enumerate(y_nodes, 1):
        a.plot([0.16, 0.91], [y, y], color=row_color, linewidth=1)
        a.text(0.12, y, rf"$V_{index}$", ha="right", va="center", color=row_color)
    for index, x in enumerate(x_nodes, 1):
        a.annotate("", (x, 0.16), (x, 0.9),
                   arrowprops={"arrowstyle": "->", "color": col_color, "lw": 1})
        a.text(x, 0.09, rf"$I_{index}$", ha="center", va="center", color=col_color)
        a.scatter([x] * 3, y_nodes, s=31, edgecolor="#233942", facecolor="#dbe8eb", zorder=3)
    a.annotate(r"$G_{22}$", (x_nodes[1], y_nodes[1]), (0.68, 0.68),
               arrowprops={"arrowstyle": "-", "color": "#233942", "lw": 0.6},
               ha="left", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

    b.plot([0.14, 0.83], [0.76, 0.76], color=row_color, linewidth=1)
    b.text(0.11, 0.76, r"$V$", ha="right", va="center", color=row_color)
    for x, sign in ((0.38, "+"), (0.68, "-")):
        b.annotate("", (x, 0.30), (x, 0.76),
                   arrowprops={"arrowstyle": "->", "color": col_color, "lw": 1})
        b.scatter(x, 0.53, s=36, edgecolor="#233942", facecolor="#dbe8eb", zorder=3)
        b.text(x + 0.045, 0.53, rf"$G^{sign}$", ha="left", va="center")
        b.text(x, 0.23, rf"$I^{sign}$", ha="center", va="center", color=col_color)
    b.text(0.52, 0.08, r"$I^+ - I^- = V(G^+ - G^-)$", ha="center", va="center")
    save(fig, directory, "fig2_crossbar")


def plot_device(directory, device):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.25, 2.85), layout="constrained")
    level = device.level_uS
    label(a, "a", "Mean relative decrease")
    a.plot(level, device.mean_decrease_percent, color="#215C8D", marker="o")
    a.set(xlabel="Nominal conductance (μS)", ylabel="Mean decrease (%)", xticks=level, ylim=(0, 5.6))

    label(b, "b", "Ratio distributions")
    b.axvline(1, linestyle="--", linewidth=0.9, color="#666666")
    b.hlines(level, device.ratio_p05, device.ratio_p95, linewidth=1.6, color="#527f87")
    b.scatter(device.ratio_mean, level, color="#174b53", s=19, zorder=3)
    b.set(xlabel="Drift-to-origin conductance ratio", ylabel="Nominal conductance (μS)", yticks=level)
    save(fig, directory, "fig3_device_ratios")


def plot_accuracy(directory, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.78), layout="constrained", sharey=True)
    for letter, (name, color), ax in zip("ab", MODELS.items(), axes, strict=True):
        subset = seeds[seeds.model == name]
        accuracy = subset[["float_percent", "quantized_percent", "endpoint_mean_percent"]].to_numpy()
        label(ax, letter, name)
        ax.plot([0, 1, 2], accuracy.T, marker=".", color=color, alpha=0.26, linewidth=0.7)
        ax.plot([0, 1, 2], accuracy.mean(axis=0), marker="o", color=color, linewidth=2.1)
        ax.set(xticks=[0, 1, 2], xticklabels=["Floating\npoint", "Quantized", "Empirical\nendpoint"], xlim=(-0.25, 2.25))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Test accuracy (%)")
    save(fig, directory, "fig4_accuracy")


def plot_losses(directory, seeds, scores):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.25, 2.84), layout="constrained")
    label(a, "a", "Interpolated response")
    label(b, "b", "Endpoint controls")
    a.axhline(0, color="#777777", linewidth=0.8)
    b.axhline(0, color="#777777", linewidth=0.8)
    for index, (name, color) in enumerate(MODELS.items()):
        subset = seeds[seeds.model == name].set_index("training_seed")
        s = scores[(scores.model == name) & (scores.condition == "empirical")]
        mean_acc = (100 * s.correct / s.test_count).groupby([s.training_seed, s.severity]).mean().unstack()
        losses = mean_acc.rsub(subset.quantized_percent, axis=0)
        losses.insert(0, 0.0, 0.0)
        mean = losses.mean()
        sd = losses.std(ddof=1)
        a.plot(mean.index, mean, color=color, linewidth=1.8, marker="o", markersize=3, label=name)
        a.fill_between(mean.index, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
        endpoints = subset[["state_mean_loss_pp", "endpoint_loss_pp"]].to_numpy()
        x = np.array([0.0, 1.0]) + (-0.075 if index == 0 else 0.075)
        b.plot(x, endpoints.T, marker=".", linewidth=0.6, color=color, alpha=0.31)
        b.scatter(x, endpoints.mean(axis=0), color=color, s=30, zorder=3, edgecolor="white", linewidth=0.5)
    a.legend(loc="upper left", frameon=False)
    a.set(xlabel="Interpolation parameter λ", ylabel="Accuracy loss (percentage points)", xlim=(-0.04, 1.04), xticks=np.linspace(0, 1, 6))
    b.set(xticks=[0, 1], xticklabels=["State means", "Empirical ratios"], xlim=(-0.32, 1.32))
    b.set_ylabel("Accuracy loss (percentage points)")
    save(fig, directory, "fig5_drift_response")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    out = args.run / "figures"
    out.mkdir(exist_ok=True)
    device = pd.read_csv(args.run / "device_summary.csv")
    seeds = pd.read_csv(args.run / "seed_summary.csv")
    scores = pd.read_csv(args.run / "trial_scores.csv")
    plot_workflow(out)
    plot_crossbar(out)
    plot_device(out, device)
    plot_accuracy(out, seeds)
    plot_losses(out, seeds, scores)


if __name__ == "__main__":
    main()
