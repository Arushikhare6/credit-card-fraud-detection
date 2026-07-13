import os
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
 
from config import PLOTS_DIR, TARGET_COL
 
 
def plot_class_imbalance(df: pd.DataFrame, save: bool = True):
    counts = df[TARGET_COL].value_counts()
    pct = 100 * counts / counts.sum()
 
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                palette=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
    ax.set_ylabel("Number of transactions")
    ax.set_title(f"Class distribution — {pct[1]:.3f}% fraud")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
 
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "class_imbalance.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fig
 
 
def plot_amount_by_class(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x=TARGET_COL, y="Amount", data=df, showfliers=False, ax=ax)
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_title("Transaction amount by class (outliers hidden)")
 
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "amount_distribution.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fig
 
 
def plot_time_by_class(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(9, 4))
    df_plot = df.copy()
    df_plot["Hour"] = (df_plot["Time"] % (24 * 3600)) // 3600
    sns.histplot(data=df_plot, x="Hour", hue=TARGET_COL, stat="density",
                 common_norm=False, bins=24, ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_title("Transaction timing (hour of day) — legit vs fraud, density-normalized")
 
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "time_distribution.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fig
 
 
def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    corr = df.drop(columns=["Time"]).corr()
 
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Feature correlation matrix (PCA components + Amount + Class)")
 
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "correlation_matrix.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return fig
 
 
def run_full_eda(df: pd.DataFrame):
    """Generate and save every EDA plot in one call."""
    plot_class_imbalance(df)
    plot_amount_by_class(df)
    plot_time_by_class(df)
    plot_correlation_heatmap(df)
    print(f"EDA plots saved to {PLOTS_DIR}")
 
 
if __name__ == "__main__":
    from data_loader import load_data
    data = load_data()
    run_full_eda(data)