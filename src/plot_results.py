"""Create report-ready plots from model comparison and Optuna outputs."""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def plot_model_comparison():
    source = RESULTS / "model_comparison.csv"
    if not source.exists():
        print("Skipping model comparison plot: results/model_comparison.csv not found.")
        return
    data = pd.read_csv(source).sort_values("macro_f1", ascending=True)
    ax = data.plot(
        x="model",
        y=["macro_f1", "precision", "recall", "top1_accuracy"],
        kind="barh",
        figsize=(10, 6),
    )
    ax.set_title("Candidate classifier comparison")
    ax.set_xlabel("Score")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    target = RESULTS / "model_comparison.png"
    plt.savefig(target, dpi=180)
    plt.close()
    print(f"Saved {target.relative_to(ROOT)}")


def plot_optuna_history():
    source = RESULTS / "optuna_trials.csv"
    if not source.exists():
        print("Skipping Optuna plot: results/optuna_trials.csv not found.")
        return
    data = pd.read_csv(source)
    if "value" not in data.columns:
        print("Skipping Optuna plot: trial file has no objective value column.")
        return
    values = pd.to_numeric(data["value"], errors="coerce").dropna()
    if values.empty:
        print("Skipping Optuna plot: no completed trial values.")
        return
    best_so_far = values.cummax()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(values) + 1), values, marker="o",
            linewidth=1, label="Trial Macro F1")
    ax.plot(range(1, len(values) + 1), best_so_far,
            linewidth=2, label="Best so far")
    ax.set_title("Optuna optimisation history")
    ax.set_xlabel("Completed trial")
    ax.set_ylabel("Grouped CV Macro F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target = RESULTS / "optuna_optimization_history.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(f"Saved {target.relative_to(ROOT)}")


def main():
    RESULTS.mkdir(exist_ok=True)
    plot_model_comparison()
    plot_optuna_history()


if __name__ == "__main__":
    main()
