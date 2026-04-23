"""Reusable EDA and reporting plotting utilities."""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np


def plot_class_distribution(df: pd.DataFrame, target: str = "Class", save_path=None):
    """Bar chart showing how many legit vs fraud transactions there are."""
    counts = df[target].value_counts()
    fraud_rate = df[target].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Legitimate", "Fraud"], counts.values, color=["steelblue", "crimson"])
    ax.set_title(f"Class Distribution  (fraud rate: {fraud_rate:.3%})")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count:,}", ha="center", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_amount_by_class(df: pd.DataFrame, amount_col: str = "Amount", target: str = "Class", save_path=None):
    """Side-by-side histograms of transaction amounts for legit vs fraud."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cls, label, colour in zip(axes, [0, 1], ["Legitimate", "Fraud"], ["steelblue", "crimson"]):
        subset = df[df[target] == cls][amount_col]
        ax.hist(subset, bins=60, color=colour, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{label}  (median: £{subset.median():.2f})")
        ax.set_xlabel("Amount (£)")
        ax.set_ylabel("Count")
        ax.axvline(subset.median(), color="black", linestyle="--", linewidth=1.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, features: list, target: str = "Class", save_path=None):
    """Grid of overlapping histograms for given features, split by class."""
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, feat in zip(axes, features):
        for cls, label, colour in [(0, "Legit", "steelblue"), (1, "Fraud", "crimson")]:
            data = df[df[target] == cls][feat].dropna()
            ax.hist(data, bins=60, alpha=0.55, color=colour, label=label, density=True)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Feature Distributions by Class", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, cols: list = None, save_path=None):
    """Correlation heatmap for selected columns."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(max(10, len(cols)), max(8, len(cols) * 0.7)))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, ax=ax, annot=len(cols) <= 15, fmt=".2f")
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_fraud_by_hour(df: pd.DataFrame, target: str = "Class", save_path=None):
    """Line chart of fraud count and fraud rate by hour of day."""
    if "hour" not in df.columns:
        df["hour"] = (df["Time"] // 3600) % 24

    by_hour = df.groupby("hour").agg(
        total=(target, "count"),
        fraud=(target, "sum"),
    )
    by_hour["rate"] = by_hour["fraud"] / by_hour["total"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.bar(by_hour.index, by_hour["fraud"], color="crimson", alpha=0.8)
    ax1.set_ylabel("Fraud Count")
    ax1.set_title("Fraud Activity by Hour of Day")

    ax2.plot(by_hour.index, by_hour["rate"] * 100, color="darkorange", linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.set_xticks(range(0, 24))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()