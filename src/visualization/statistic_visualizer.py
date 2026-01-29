import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
import numpy as np

from typing import final, Set, List, Dict, Any, Tuple, cast, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.container import BarContainer
from numpy.typing import NDArray

import src.visualization.config as cfg
from src.visualization.base_visualizer import BaseVisualizer


@final
class StatisticVisualizer(BaseVisualizer):
    def __init__(self, dataset_path: str, df_path: str = ""):
        super().__init__(dataset_path, df_path)

        self.known_classes: Set[str] = {
            "glioma",
            "meningioma",
            "no_tumor",
            "pituitary",
            "normal",
            "tumor",
            "images",
            "mask",
        }

        self.computed_paths: List[Path] = []

    def compute_images_per_folder(
        self, look_at: str, save_dataframe: bool = False
    ) -> None:
        if self.dataframe is None:
            paths = [
                p
                for p in self.dataset_path.rglob("*")
                if p.is_dir()
                and look_at in str(p)
                and any(child.is_file() for child in p.iterdir())
            ]

            d: Dict[str, List[Any]] = {"Path": []}

            if "raw" in str(self.dataset_path):
                for path in paths:
                    d["Path"].append(path.name)
                    if d.get("Count") is None:
                        d["Count"] = []
                    d["Count"].append(len(os.listdir(path)))

            else:
                for path in paths:
                    self.computed_paths.append(path)

                    pn: str = path.parent.name
                    cn: str = path.name

                    if pn not in self.known_classes:
                        pn, cn = cn, pn

                    if cn not in d["Path"]:
                        d["Path"].append(cn)

                    if d.get(pn) is None:
                        d[pn] = []

                    index: int = d["Path"].index(cn)
                    d[pn].insert(index, len(os.listdir(path)))

            self.dataframe = pd.DataFrame(d)
            self.dataframe = self.dataframe.set_index("Path")

            if save_dataframe:
                self.dataframe.to_csv("dataframe.csv", index=False)
            else:
                print(self.dataframe)

    def plot_bar_chart(
        self, log_scale: bool = False, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        """
        Plot a bar chart

        return fig, axg the number of samples per class and split.

        Args:
            log_scale (bool): Whether to use logarithmic y-axis.
            save_path (str | None): Path to save the figure. If None, shows

        return fig, ax of the plot.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        self.dataframe.T.plot(
            kind="bar", rot=cfg.LABEL_ROTATION, width=0.85, ax=ax, edgecolor="black"
        )

        ax.set_title(
            "Number of Samples per Class" + (" (Log Scale)" if log_scale else ""),
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Images")

        if log_scale:
            ax.set_yscale("log")

        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        for p in ax.patches:
            r = cast(Rectangle, p)
            height = r.get_height()
            if height > 0:
                ax.annotate(
                    text=f"{int(height):,}",
                    xy=(r.get_x() + r.get_width() / 2, height),
                    ha=cfg.ANNOTATE_HA,
                    va=cfg.ANNOTATE_VA,
                    fontsize=cfg.TITLE_FONTSIZE,
                    xytext=cfg.ANNOTATE_XYTEXT,
                    textcoords=cfg.ANNOTATE_TEXTCOORDS,
                )

        ax.legend(
            title="Split",
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_class_imbalance_ratio(self, save_path: str = "") -> Tuple[Figure, Axes]:
        """
        Plot class imbalance ratio per split.
        Imbalance ratio is defined as:
            max_class_count / class_count
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        # Avoid division by zero
        df_copy: pd.DataFrame = df_copy.replace(0, np.nan)

        max_counts: pd.Series = df_copy.max(axis=1)
        ratio_df: pd.DataFrame = df_copy.div(max_counts, axis=0)

        plot_df: pd.DataFrame = (
            ratio_df.reset_index()
            .melt(id_vars="Path", var_name="Class", value_name="ImbalanceRatio")
            .dropna()
        )

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        sns.barplot(
            data=plot_df,
            x="Class",
            y="ImbalanceRatio",
            hue="Path",
            palette="colorblind",
            ax=ax,
        )

        ax.set_title(
            "Class Imbalance Ratio (Relative to Largest Class)",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Imbalance Ratio")

        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(
                    container, fmt="%.2f", padding=3, fontsize=cfg.TITLE_FONTSIZE
                )

        ax.legend(
            title="Split",
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_split_distribution(
        self, normalize: bool = True, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        """
        Plot class distribution per dataset split.

        Args:
            normalize (bool): If True, plot percentages instead of raw counts.
            save_path (str | None): Path to save the figure.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        if normalize:
            row_sums: pd.Series = df_copy.sum(axis=1)
            df_copy: pd.DataFrame = (
                df_copy.div(row_sums.replace(0, np.nan), axis=0) * 100
            )

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        df_copy.plot(kind="bar", stacked=True, rot=cfg.LABEL_ROTATION, ax=ax)

        plt.title(
            "Class Distribution per Dataset Split",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Split")
        plt.ylabel("Percentage (%)" if normalize else "Number of samples")

        if normalize:
            ax.set_ylim(0, 100)

        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        plt.legend(
            title="Class",
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_heatmap(
        self, normalize: bool = True, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        """
        Plot a heatmap of class distribution per dataset split.

        Args:
            normalize (bool): If True, show p

        return fig, axercentages per split.
            save_path (str | None): Path to save the figure.
            :param normalize:
            :param save_path:
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        if normalize:
            row_sums: pd.Series = df_copy.sum(axis=1).replace(0, np.nan)
            df_copy: pd.DataFrame = df_copy.div(row_sums, axis=0) * 100
            fmt: str = ".2f"
            cbar_kws: Dict = {"label": "Percentage (%)"}
        else:
            fmt: str = ".0f"
            cbar_kws: Dict = {"label": "Number of Images"}

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        sns.heatmap(
            df_copy,
            annot=True,
            fmt=fmt,
            cmap="viridis",
            cbar_kws=cbar_kws,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )

        ax.set_title(
            "Class Distribution per Folder",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Folder")

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_stacked_bar_chart(
        self, normalize: bool = True, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        """
        Plot stacked bar chart of class distribution per dataset split.

        Args:
            normalize (bool): If True, plot percentages per split.
            save_path (str | None): Path to save the figure.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        if normalize:
            row_sums: pd.Series = df_copy.sum(axis=1).replace(0, np.nan)
            df_copy: pd.DataFrame = df_copy.div(row_sums, axis=0) * 100

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        df_copy.plot(
            kind="bar", stacked=True, colormap="tab20", rot=cfg.LABEL_ROTATION, ax=ax
        )

        ax.set_title(
            "Class Distribution per Folder",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Folder")
        ax.set_ylabel("Percentage (%)" if normalize else "Number of Images")
        ax.legend(
            title="Class",
            frameon=cfg.LEGEND_FRAME,
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
        )

        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_dominant_class_ratio(self, save_path: str = "") -> Tuple[Figure, Axes]:
        """
        Plot dominant class ratio (class purity) per dataset split.

        Dominant class ratio is defined as:
            max_class_count / total_count
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        totals: pd.Series = df_copy.sum(axis=1).replace(0, np.nan)
        purity: pd.Series = df_copy.max(axis=1) / totals

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        ax.bar(df_copy.index, purity)

        ax.set_ylim(0, 1)

        ax.set_title(
            "Dominant Class Ratio per Folder",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Folder")
        ax.set_ylabel("Dominant Class Ratio")

        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        for x, y in zip(df_copy.index, purity):
            if not np.isnan(y):
                ax.annotate(
                    text=f"{y:.2f}",
                    xy=(x, y),
                    ha=cfg.ANNOTATE_HA,
                    va=cfg.ANNOTATE_VA,
                    fontsize=cfg.TITLE_FONTSIZE,
                    xytext=cfg.ANNOTATE_XYTEXT,
                    textcoords=cfg.ANNOTATE_TEXTCOORDS,
                )

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_folder_entropy(
        self, normalize: bool = True, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        """
        Plot class entropy per dataset split.

        Entropy is computed over class probabilities per split.
        If normalize=True, entropy is normalized to [0, 1].
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        totals: pd.Series = df_copy.sum(axis=1).replace(0, np.nan)
        probs: pd.DataFrame = df_copy.div(totals, axis=0)

        ent: pd.Series = -(probs * np.log2(probs.replace(0, np.nan))).sum(axis=1)

        if normalize:
            max_ent = np.log(df_copy.shape[1])
            ent = ent / max_ent

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        ax.plot(df_copy.index, ent, marker="o", linestyle="-")

        ax.set_title(
            "Class Entropy per Folder",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Folder")
        ax.set_ylabel("Normalized Entropy" if normalize else "Entropy")

        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        for x, y in zip(df_copy.index, ent):
            if not np.isnan(y):
                ax.annotate(
                    text=f"{y:.2f}",
                    xy=(x, y),
                    ha=cfg.ANNOTATE_HA,
                    va=cfg.ANNOTATE_VA,
                    fontsize=cfg.TITLE_FONTSIZE,
                    xytext=cfg.ANNOTATE_XYTEXT,
                    textcoords=cfg.ANNOTATE_TEXTCOORDS,
                )

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_gini_index_per_class(self, save_path: str = "") -> Tuple[Figure, Axes]:
        """
        Plot Gini index per class, measuring distribution inequality across dataset splits.

        Gini index = 0 means perfect equality across splits.
        Gini index = 1 means maximal inequality (all samples in one split).
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        def gini(x: Union[pd.Series, NDArray[np.floating]]) -> float:
            x1: NDArray[np.float64] = np.asarray(x, dtype=float)
            if np.sum(x1) == 0:
                return float("nan")  # avoid division by zero
            return float(np.sum(np.abs(x1[:, None] - x1)) / (2 * len(x1) * np.sum(x1)))

        classes: List[str] = list(df_copy.columns)
        gini_vals: NDArray[np.float64] = np.array(
            [
                gini(
                    df_copy[c] / df_copy[c].sum()
                    if df_copy[c].sum() > 0
                    else np.array([0])
                )
                for c in classes
            ],
            dtype=float,
        )

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        ax.bar(classes, gini_vals, color=cfg.DEFAULT_BAR_COLOR)
        ax.set_title(
            "Gini Index per Class (distribution across folders)",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Gini Index")
        ax.set_ylim(0, 1)

        # value labels
        for i, v in enumerate(gini_vals):
            if isinstance(v, (np.ndarray, tuple, list)):
                v = float(v[0])
            elif np.isnan(v):
                continue

            ax.annotate(
                text=f"{v:.2f}",
                xy=(i, v),
                ha=cfg.ANNOTATE_HA,
                va=cfg.ANNOTATE_VA,
                fontsize=cfg.ANNOTATE_FONTSIZE,
                xytext=cfg.ANNOTATE_XYTEXT,
                textcoords=cfg.ANNOTATE_TEXTCOORDS,
            )

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_cumulative_image_count(self, save_path: str = "") -> Tuple[Figure, Axes]:
        """
        Plot cumulative image count per class across dataset splits/folders.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()
        cum: pd.DataFrame = df_copy.cumsum()
        classes: List[str] = list(df_copy.columns)

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        palette = sns.color_palette(cfg.PALETTE_TAB10, n_colors=len(classes))

        for i, c in enumerate(classes):
            ax.plot(cum[c], marker="o", label=c, linewidth=2, color=palette[i])

        ax.set_xticks(range(len(df_copy)))

        ax.set_title(
            "Cumulative Image Count per Class Across Folders",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_ylabel("Cumulative Image Count")
        ax.set_xlabel("Folder")

        ax.legend(
            title="Class",
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
            frameon=cfg.LEGEND_FRAME,
        )
        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_box_plot_image_count(self, save_path: str = "") -> Tuple[Figure, Axes]:
        """
        Plot distribution of images per class across dataset splits.

        Combines boxplot (summary statistics) and swarmplot (individual points).
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()
        classes: List[str] = list(df_copy.columns)

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

        # Boxplot for summary
        sns.boxplot(data=df_copy[classes], palette=cfg.PALETTE_PASTEL, ax=ax)

        # Swarmplot for individual points
        # Reduce size for readability in large datasets
        sns.swarmplot(data=df_copy[classes], color=".25", size=5, ax=ax)

        ax.set_title(
            "Images per Folder per Class",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_ylabel("Number of Images")
        ax.set_xlabel("Class")
        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        self._save_figure(fig, save_path)

        return fig, ax
