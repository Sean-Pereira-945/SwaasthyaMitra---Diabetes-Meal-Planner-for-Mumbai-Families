from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_bar_chart(summary: pd.DataFrame, x_col: str, y_col: str, title: str, output_png: Path, ylim=None) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=summary, x=x_col, y=y_col, palette="Set2")
    ax.set_title(title)
    ax.set_xlabel("System")
    ax.set_ylabel(y_col.replace("_", " ").title())
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def build_all_charts(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {"case_type", "gold_accuracy", "adherence_score", "safety_score", "latency_sec"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV must include columns: {sorted(required)}. Missing: {missing}")

    metric_cols = ["gold_accuracy", "adherence_score", "safety_score", "latency_sec"]
    optional_cols = [col for col in ["hard_fail_count", "medically_safe"] if col in df.columns]
    summary = df.groupby("case_type", as_index=False)[metric_cols + optional_cols].mean().sort_values("gold_accuracy", ascending=False)
    summary.to_csv(output_dir / "metric_summary.csv", index=False)

    _save_bar_chart(
        summary=summary,
        x_col="case_type",
        y_col="gold_accuracy",
        title="Average Nutritional Accuracy by System",
        output_png=output_dir / "accuracy_comparison.png",
        ylim=(0, 1),
    )

    _save_bar_chart(
        summary=summary,
        x_col="case_type",
        y_col="adherence_score",
        title="Estimated Adherence by System",
        output_png=output_dir / "adherence_comparison.png",
        ylim=(0, 1),
    )

    _save_bar_chart(
        summary=summary,
        x_col="case_type",
        y_col="safety_score",
        title="Safety Score by System",
        output_png=output_dir / "safety_comparison.png",
        ylim=(0, 1),
    )

    _save_bar_chart(
        summary=summary,
        x_col="case_type",
        y_col="latency_sec",
        title="Average Response Latency (Seconds)",
        output_png=output_dir / "latency_comparison.png",
        ylim=None,
    )

    if "hard_fail_count" in summary.columns:
        _save_bar_chart(
            summary=summary,
            x_col="case_type",
            y_col="hard_fail_count",
            title="Average Strict Rule Failures",
            output_png=output_dir / "hard_fail_comparison.png",
            ylim=None,
        )

    if "medically_safe" in summary.columns:
        _save_bar_chart(
            summary=summary,
            x_col="case_type",
            y_col="medically_safe",
            title="Medical Safety Pass Rate",
            output_png=output_dir / "medical_safety_pass_rate.png",
            ylim=(0, 1),
        )

    # Case type breakdown: RAG gain over baseline where paired IDs exist.
    if "case_id" in df.columns:
        pivot = df.pivot_table(index="case_id", columns="case_type", values="gold_accuracy", aggfunc="mean")
        if "rag" in pivot.columns and "baseline" in pivot.columns:
            pivot["delta_accuracy"] = pivot["rag"] - pivot["baseline"]
            pivot.to_csv(output_dir / "paired_accuracy_delta.csv")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    input_csv = root / "test_cases.csv"
    output_dir = root

    build_all_charts(input_csv, output_dir)
    print(f"Saved charts and summaries to: {output_dir}")
