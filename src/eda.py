"""
Exploratory Data Analysis visualizations for the bank marketing dataset.

Usage
-----
    from src.eda import (
        plot_target_distribution,
        plot_numeric_distributions,
        plot_outlier_boxplots,
        plot_correlation_heatmap,
        plot_categorical_vs_target,
        plot_feature_importance,
    )
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_target_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Count plot of term deposit subscription (yes / no)."""
    plt.figure(figsize=(6, 4))
    df["term_deposit"].value_counts().plot(kind="bar", color=["steelblue", "coral"], edgecolor="white")
    plt.title("Term Deposit Subscription Distribution")
    plt.xlabel("Subscribed")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_numeric_distributions(df: pd.DataFrame, cols: list[str], save_path: str | None = None) -> None:
    """Distribution plots (with skew + kurtosis) for numeric columns."""
    n = len(cols)
    fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3, figsize=(15, 4 * ((n + 2) // 3)))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], bins=25, kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"{col}\nskew={df[col].skew():.2f}  kurt={df[col].kurt():.2f}")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_outlier_boxplots(df: pd.DataFrame, cols: list[str], save_path: str | None = None) -> None:
    """Boxplots to visualise outliers across numeric columns."""
    n = len(cols)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(col, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Outlier Boxplots — Numeric Features", fontsize=13)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Heatmap of correlations among numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr,
        mask=np.zeros_like(corr, dtype=bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_categorical_vs_target(
    df: pd.DataFrame,
    cat_cols: list[str],
    target: str = "term_deposit",
    save_path: str | None = None,
) -> None:
    """Bar charts of category counts for customers who subscribed (term_deposit=yes)."""
    subscribed = df[df[target] == "yes"]
    n = len(cat_cols)
    fig, axes = plt.subplots(nrows=2, ncols=(n + 1) // 2, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        subscribed[col].value_counts().plot.bar(ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis="x", rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Categorical Feature Distribution — Subscribed Customers", fontsize=13)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: list[str],
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of mutual information or model feature importances."""
    s = pd.Series(importance, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.3)))
    s.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Feature Importance (Mutual Information)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = "Model",
    save_path: str | None = None,
) -> None:
    """ROC curve plot."""
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "r--", label="Random baseline")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    _save_or_show(save_path)


def _save_or_show(save_path: str | None) -> None:
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()
