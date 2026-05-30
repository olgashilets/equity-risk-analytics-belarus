# -*- coding: utf-8 -*-
"""Asset clustering utilities for Belarusian equities.

The clustering stage intentionally uses monthly simple returns. This keeps the
asset universe selection reproducible and separated from the downstream CVaR
portfolio module, where log returns are used for portfolio risk calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

ALL_STOCKS = [
    "СберБанк", "Приорбанк", "МАПИД", "Трест-35",
    "СветлогорскЗЖБиИК", "Минскпромстрой", "Брестгазоаппарат",
    "Керамин", "БЭРН", "Белинвестбанк", "БрестМК",
    "МозырьНПЗ", "СРСУ-3", "СлуцкСахарКомб", "ЦУМ Минск",
    "Молочный Мир",
]


@dataclass(frozen=True)
class ClusteringConfig:
    min_months: int = 10
    threshold: float = 1.2
    n_clusters: int = 7


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a wide monthly price table and return it indexed by date."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    date_col = "Дата" if "Дата" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError("Input data must contain a date column named 'Дата' or 'date'.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).sort_values("date").set_index("date")
    return df


def prepare_prices(prices: pd.DataFrame, config: ClusteringConfig) -> tuple[pd.DataFrame, list[str]]:
    """Filter assets and fill missing monthly prices.

    Zero values are treated as missing prices. Assets with fewer than
    ``config.min_months`` non-missing observations are excluded.
    """
    available = [stock for stock in ALL_STOCKS if stock in prices.columns]
    if not available:
        raise ValueError("No expected asset columns were found in the input data.")

    selected = prices[available].copy().replace(0, np.nan)
    good_stocks = [col for col in selected.columns if selected[col].notna().sum() >= config.min_months]
    if not good_stocks:
        raise ValueError(f"No assets have at least {config.min_months} monthly observations.")

    clean = selected[good_stocks].ffill().interpolate(method="linear", limit_direction="both").bfill()
    clean = clean.dropna(axis=1, how="any")
    return clean, list(clean.columns)


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly simple returns using pct_change()."""
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return returns.fillna(0.0)


def annual_return_risk(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate annualized return and volatility in percent."""
    return pd.DataFrame({
        "asset": returns.columns,
        "annual_return_pct": (returns.mean() * 12 * 100).round(1).values,
        "annual_risk_pct": (returns.std(ddof=1) * np.sqrt(12) * 100).round(1).values,
    })


def run_clustering(prices: pd.DataFrame, config: ClusteringConfig = ClusteringConfig()) -> dict[str, object]:
    """Run the clustering workflow and return intermediate results."""
    clean_prices, good_stocks = prepare_prices(prices, config)
    returns = compute_simple_returns(clean_prices)

    corr = returns.corr().fillna(0.0)
    distance_matrix = 1.0 - corr.abs()
    np.fill_diagonal(distance_matrix.values, 0.0)

    # Legacy-compatible implementation used in the original project: Ward linkage
    # is applied to the rows of the distance matrix. This reproduces the published
    # selected universe.
    condensed = pdist(distance_matrix.values, metric="euclidean")
    link = linkage(condensed, method="ward")
    clusters = fcluster(link, t=config.threshold, criterion="distance")
    if len(set(clusters)) != config.n_clusters:
        clusters = fcluster(link, t=config.n_clusters, criterion="maxclust")

    stats = annual_return_risk(returns)
    cluster_table = stats.copy()
    cluster_table["cluster_id"] = clusters
    cluster_table = cluster_table[["asset", "cluster_id", "annual_return_pct", "annual_risk_pct"]]
    cluster_table = cluster_table.sort_values(["cluster_id", "annual_return_pct"], ascending=[True, False]).reset_index(drop=True)

    selected_rows = []
    for cluster_id in sorted(cluster_table["cluster_id"].unique()):
        cluster_assets = cluster_table[cluster_table["cluster_id"] == cluster_id]
        best = cluster_assets.sort_values(["annual_return_pct", "annual_risk_pct"], ascending=[False, True]).iloc[0]
        selected_rows.append(best)
    selected_assets = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected_assets["selection_rule"] = "highest annualized simple return within cluster"

    return {
        "clean_prices": clean_prices,
        "returns": returns,
        "correlation": corr,
        "distance_matrix": distance_matrix,
        "linkage": link,
        "cluster_table": cluster_table,
        "selected_assets": selected_assets,
        "good_stocks": good_stocks,
    }


def save_outputs(result: dict[str, object], output_dir: str | Path) -> None:
    """Save tables and figures for the clustering case study."""
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    clean_prices: pd.DataFrame = result["clean_prices"]  # type: ignore[assignment]
    returns: pd.DataFrame = result["returns"]  # type: ignore[assignment]
    corr: pd.DataFrame = result["correlation"]  # type: ignore[assignment]
    cluster_table: pd.DataFrame = result["cluster_table"]  # type: ignore[assignment]
    selected_assets: pd.DataFrame = result["selected_assets"]  # type: ignore[assignment]
    link = result["linkage"]

    clean_prices.to_csv(tables / "clean_prices.csv", encoding="utf-8-sig")
    returns.to_csv(tables / "simple_returns.csv", encoding="utf-8-sig")
    corr.to_csv(tables / "correlation_matrix.csv", encoding="utf-8-sig")
    cluster_table.to_csv(tables / "cluster_results.csv", index=False, encoding="utf-8-sig")
    selected_assets.to_csv(tables / "selected_assets.csv", index=False, encoding="utf-8-sig")

    selected_names = selected_assets["asset"].tolist()
    selected_price_matrix = clean_prices[selected_names].copy()
    selected_price_matrix.to_csv(tables / "selected_asset_prices.csv", encoding="utf-8-sig")
    selected_price_matrix.reset_index().melt(
        id_vars="date", var_name="asset", value_name="price"
    ).to_csv(tables / "selected_asset_prices_long.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    dendrogram(
        link,
        labels=list(clean_prices.columns),
        ax=ax1,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=ClusteringConfig().threshold,
    )
    ax1.set_title("Hierarchical clustering dendrogram")
    ax1.set_xlabel("Assets")
    ax1.set_ylabel("Distance")
    ax1.axhline(y=ClusteringConfig().threshold, linestyle="--", linewidth=1.2)

    ax2 = plt.subplot(2, 2, 2)
    sns.heatmap(corr, annot=False, cmap="RdYlGn", center=0, square=True, linewidths=0.4, ax=ax2)
    ax2.set_title("Simple-return correlation matrix")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=7)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=7)

    ax3 = plt.subplot(2, 2, 3)
    norm_prices = clean_prices / clean_prices.iloc[0] * 100
    norm_prices.plot(ax=ax3, legend=False, alpha=0.75)
    ax3.set_title("Normalized price dynamics, start = 100")
    ax3.set_ylabel("Index")
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    scatter = ax4.scatter(
        cluster_table["annual_risk_pct"],
        cluster_table["annual_return_pct"],
        c=cluster_table["cluster_id"],
        cmap="tab20",
        s=140,
        alpha=0.75,
    )
    for _, row in cluster_table.iterrows():
        if row["asset"] in selected_names:
            ax4.annotate(row["asset"], (row["annual_risk_pct"], row["annual_return_pct"]), fontsize=8)
    ax4.set_xlabel("Annualized risk, %")
    ax4.set_ylabel("Annualized return, %")
    ax4.set_title("Return vs risk by cluster")
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label="Cluster")

    fig.tight_layout()
    fig.savefig(figures / "clustering_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_interpretation(result: dict[str, object]) -> str:
    cluster_table: pd.DataFrame = result["cluster_table"]  # type: ignore[assignment]
    selected_assets: pd.DataFrame = result["selected_assets"]  # type: ignore[assignment]
    reduction_pct = (1 - len(selected_assets) / len(cluster_table)) * 100
    selected = ", ".join(selected_assets["asset"].tolist())
    return (
        "Asset clustering interpretation\n"
        "================================\n\n"
        "The clustering stage uses monthly simple returns and correlation-based distance. "
        "It reduces the analyzed universe from "
        f"{len(cluster_table)} assets to {len(selected_assets)} cluster representatives "
        f"({reduction_pct:.0f}% reduction).\n\n"
        f"Selected assets: {selected}.\n\n"
        "This module is used as an asset-universe reduction step. Portfolio construction "
        "is intentionally handled in the separate CVaR module, where the selected asset "
        "price matrix is used as the input.\n"
    )
