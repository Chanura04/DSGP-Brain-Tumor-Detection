import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
import numpy as np
from PIL import Image
import cv2

from typing import final, Optional, Set, List, Dict, Any, Tuple, cast, Union, Iterable
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle
from matplotlib.container import BarContainer
from numpy.typing import NDArray

import visualization.config as cfg


@final
class Visualizer:
    def __init__(self, dataset_path: str, df_path: str = "", img_df_path: str = ""):
        self.dataset_path: Path = Path(dataset_path)
        self.df: Optional[pd.DataFrame] = None
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
        self.img_df: Optional[pd.DataFrame] = None

        self.df_path: Path = Path(df_path)
        self.img_df_path: Path = Path(img_df_path)

        self._read_files()

    def _read_files(self) -> None:
        if self.df_path.name != "" and self.df_path.exists():
            self.df = pd.read_csv(self.df_path, index_col="Path")

        if self.img_df_path.name != "" and self.img_df_path.exists():
            self.img_df = pd.read_csv(self.img_df_path)

    def compute_images_per_folder(
        self, look_at: str, save_dataframe: bool = False
    ) -> None:
        if self.df is None:
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

            self.df = pd.DataFrame(d)
            self.df = self.df.set_index("Path")

            if save_dataframe:
                self.df.to_csv("dataframe.csv", index=False)
            else:
                print(self.df)

    def plot_bar_chart(
        self, log_scale: bool = False, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot a bar chart

        return fig, axg the number of samples per class and split.

        Args:
            log_scale (bool): Whether to use logarithmic y-axis.
            save_path (str | None): Path to save the figure. If None, shows

        return fig, ax of the plot.
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

        self.df.T.plot(
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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_class_imbalance_ratio(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot class imbalance ratio per split.
        Imbalance ratio is defined as:
            max_class_count / class_count
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()

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
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_split_distribution(
        self, normalize: bool = True, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot class distribution per dataset split.

        Args:
            normalize (bool): If True, plot percentages instead of raw counts.
            save_path (str | None): Path to save the figure.
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()

        if normalize:
            row_sums: pd.Series = df_copy.sum(axis=1)
            df_copy: pd.DataFrame = (
                df_copy.div(row_sums.replace(0, np.nan), axis=0) * 100
            )

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_heatmap(
        self, normalize: bool = True, save_path: Optional[str] = None
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

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()

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
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )
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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_stacked_bar_chart(
        self, normalize: bool = True, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot stacked bar chart of class distribution per dataset split.

        Args:
            normalize (bool): If True, plot percentages per split.
            save_path (str | None): Path to save the figure.
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()

        if normalize:
            row_sums: pd.Series = df_copy.sum(axis=1).replace(0, np.nan)
            df_copy: pd.DataFrame = df_copy.div(row_sums, axis=0) * 100

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )
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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_dominant_class_ratio(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot dominant class ratio (class purity) per dataset split.

        Dominant class ratio is defined as:
            max_class_count / total_count
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()

        totals: pd.Series = df_copy.sum(axis=1).replace(0, np.nan)
        purity: pd.Series = df_copy.max(axis=1) / totals

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_folder_entropy(
        self, normalize: bool = True, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot class entropy per dataset split.

        Entropy is computed over class probabilities per split.
        If normalize=True, entropy is normalized to [0, 1].
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()

        totals: pd.Series = df_copy.sum(axis=1).replace(0, np.nan)
        probs: pd.DataFrame = df_copy.div(totals, axis=0)

        ent: pd.Series = -(probs * np.log2(probs.replace(0, np.nan))).sum(axis=1)

        if normalize:
            max_ent = np.log(df_copy.shape[1])
            ent = ent / max_ent

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_gini_index_per_class(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot Gini index per class, measuring distribution inequality across dataset splits.

        Gini index = 0 means perfect equality across splits.
        Gini index = 1 means maximal inequality (all samples in one split).
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()

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
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )
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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_cumulative_image_count(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot cumulative image count per class across dataset splits/folders.
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()
        cum: pd.DataFrame = df_copy.cumsum()
        classes: List[str] = list(df_copy.columns)

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

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

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_box_plot_image_count(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot distribution of images per class across dataset splits.

        Combines boxplot (summary statistics) and swarmplot (individual points).
        """

        if self.df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_images_per_folder() first."
            )

        df_copy: pd.DataFrame = self.df.copy()
        classes: List[str] = list(df_copy.columns)

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

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

        # Optional save
        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def compute_image_analysis(self, save_dataframe=False):
        if self.img_df is None:
            data: List[Dict] = []

            for path in self.computed_paths:
                p = path.parent.name
                c = path.name

                if p not in self.known_classes:
                    p, c = c, p

                for img_path in path.glob("*"):
                    img = Image.open(img_path).convert("RGB")
                    arr = np.array(img)

                    # features
                    file_size_kb = img_path.stat().st_size / 1024

                    # Laplacian variance for sharpness
                    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                    sharpness = cv2.Laplacian(
                        gray, cv2.CV_64F
                    ).var()  # simple proxy for sharpness
                    R: float = arr[:, :, 0].mean()
                    G: float = arr[:, :, 1].mean()
                    B: float = arr[:, :, 2].mean()
                    brightness: float = arr.mean()
                    contrast: float = arr.std()

                    data.append(
                        {
                            "Class": p,
                            "FileSizeKB": file_size_kb,
                            "Sharpness": sharpness,
                            "R": R,
                            "G": G,
                            "B": B,
                            "Brightness": brightness,
                            "Contrast": contrast,
                        }
                    )

            self.img_df = pd.DataFrame(data)

            if save_dataframe:
                self.img_df.to_csv("Image_df.csv", index=False)
            else:
                print(self.img_df)

    def plot_color_channel_distribution(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot RGB color channel distributions per class using KDE plots.
        """

        if self.img_df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        channels: List[str] = ["R", "G", "B"]
        classes: List[str] = self.img_df["Class"].unique().tolist()
        palette = sns.color_palette(cfg.PALETTE_TAB10, n_colors=len(classes))

        fig: Figure
        axes: np.ndarray
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=cfg.FIGSIZE_MEDIUM,
            constrained_layout=cfg.CONSTRAINED_LAYOUT,
            sharey=True,
        )

        for i, channel in enumerate(channels):
            ax: Axes = axes[i]
            for cls, color in zip(classes, palette):
                subset: pd.DataFrame = self.img_df[self.img_df["Class"] == cls]
                if subset[channel].nunique() > 1:  # avoid zero-variance KDE warning
                    sns.kdeplot(
                        x=subset[channel].to_numpy(), label=cls, ax=ax, color=color
                    )
            ax.set_title(f"{channel} Channel Distribution")
            ax.set_xlabel("Pixel Intensity")
            if i == 0:
                ax.set_ylabel("Density")

        # Single legend outside the plots
        handles: list[Artist]  # Usually Patch or Line2D
        labels: list[str]
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Class",
            bbox_to_anchor=(0.5, -0.15),
            loc="lower center",
            ncol=len(classes),
        )

        plt.suptitle(
            "RGB Color Distribution per Class", fontsize=cfg.TITLE_FONTSIZE, y=1.05
        )

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, axes

    def plot_mean_rgb_values(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot mean RGB values per class as a grouped bar chart.
        """

        if self.img_df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        mean_vals = self.img_df.groupby("Class")[["R", "G", "B"]].mean()

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

        # Plot with colors fixed to R/G/B
        mean_vals.plot(
            kind="bar",
            color=[cfg.COLOR_R, cfg.COLOR_G, cfg.COLOR_B],
            rot=cfg.LABEL_ROTATION,
            ax=ax,
        )

        ax.set_ylabel("Mean Channel Value")
        ax.set_title(
            "Mean RGB Values per Class",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.legend(
            title="Channel",
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
            frameon=cfg.LEGEND_FRAME,
        )
        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        # Annotate bar heights
        for p in ax.patches:
            r = cast(Rectangle, p)
            if not np.isnan(r.get_height()):
                ax.annotate(
                    text=f"{r.get_height():.1f}",
                    xy=(r.get_x() + r.get_width() / 2.0, r.get_height()),
                    ha=cfg.ANNOTATE_HA,
                    va=cfg.ANNOTATE_VA,
                    fontsize=cfg.TITLE_FONTSIZE,
                    xytext=cfg.ANNOTATE_XYTEXT,
                    textcoords=cfg.ANNOTATE_TEXTCOORDS,
                )

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_image_sharpness(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot image sharpness per class using Laplacian variance.
        Log-scaled y-axis is used due to wide range of sharpness values.
        """

        if self.img_df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        df_copy: pd.DataFrame = self.img_df.copy()

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

        # Avoid zero or negative values for log scale
        df_copy["Sharpness_safe"] = df_copy["Sharpness"].clip(lower=cfg.EPSILON)

        sns.boxplot(x="Class", y="Sharpness_safe", data=df_copy, ax=ax)

        ax.set_yscale("log")
        ax.set_title(
            "Image Sharpness per Class",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_ylabel("Sharpness (Laplacian Variance)")
        ax.set_xlabel("Class")
        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        # Annotate median values
        classes: List[str] = df_copy["Class"].unique().tolist()
        for i, cls in enumerate(classes):
            median_val = df_copy[df_copy["Class"] == cls]["Sharpness"].median()
            if not np.isnan(median_val):
                ax.annotate(
                    text=f"{median_val:.1f}",
                    xy=(i, median_val),
                    ha=cfg.ANNOTATE_HA,
                    va=cfg.ANNOTATE_VA,
                    fontsize=cfg.TITLE_FONTSIZE,
                    xytext=cfg.ANNOTATE_XYTEXT,
                    textcoords=cfg.ANNOTATE_TEXTCOORDS,
                )

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_image_brightness_vs_sharpness(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Scatterplot of Brightness vs Sharpness per class.
        Optionally includes trend lines per class.
        """

        if self.img_df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        df_copy: pd.DataFrame = self.img_df.copy()
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

        # Avoid zero sharpness for log scale
        df_copy["Sharpness_safe"] = df_copy["Sharpness"].clip(lower=cfg.EPSILON)

        sns.scatterplot(
            x="Brightness",
            y="Sharpness_safe",
            hue="Class",
            data=df_copy,
            palette=cfg.PALETTE_TAB10,
            s=cfg.SCATTER_MARKER_SIZE,
            alpha=cfg.SCATTER_ALPHA,
            ax=ax,
        )

        # Optional: trend lines per class
        classes: List[str] = df_copy["Class"].unique().tolist()
        for cls in classes:
            subset = df_copy[df_copy["Class"] == cls]
            if len(subset) > 1:
                sns.regplot(
                    x="Brightness",
                    y="Sharpness_safe",
                    data=subset,
                    scatter=False,
                    ax=ax,
                    color=sns.color_palette(cfg.PALETTE_TAB10)[
                        list(classes).index(cls)
                    ],
                )

        ax.set_title(
            "Brightness vs Sharpness per Class",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_xlabel("Brightness")
        ax.set_ylabel("Sharpness (Laplacian Variance, log scale)")
        ax.set_yscale("log")
        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)
        ax.legend(
            title="Class",
            bbox_to_anchor=cfg.LEGEND_BBOX_TO_ANCHOR,
            loc=cfg.LEGEND_LOCATION,
            frameon=cfg.LEGEND_FRAME,
        )

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax

    def plot_file_distribution(
        self, save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot file size distribution per class using a log-scaled boxplot.
        Median values are annotated for clarity.
        """

        if self.img_df is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        df_copy: pd.DataFrame = self.img_df.copy()

        # Avoid zero values for log scale
        df_copy["FileSizeKB_safe"] = df_copy["FileSizeKB"].clip(lower=cfg.FILESIZE)

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

        sns.boxplot(
            x="Class",
            y="FileSizeKB_safe",
            hue="Class",
            data=df_copy,
            ax=ax,
            palette=cfg.PALETTE_PASTEL,
            dodge=False,
        )
        ax.set_yscale("log")
        ax.set_title(
            "File Size Distribution per Class",
            fontsize=cfg.TITLE_FONTSIZE,
            weight=cfg.TITLE_WEIGHT,
        )
        ax.set_ylabel("File Size (KB)")
        ax.set_xlabel("Class")
        ax.grid(axis=cfg.GRID_AXIS, linestyle=cfg.GRID_LINESTYLE, alpha=cfg.GRID_ALPHA)

        # Annotate median values
        x_positions: Iterable[int] = range(len(df_copy["Class"].unique()))
        medians: NDArray[np.float64] = (
            df_copy.groupby("Class")["FileSizeKB_safe"]
            .median()
            .to_numpy(dtype=np.float64)
        )
        for x, y in zip(x_positions, medians):
            if not np.isnan(y):
                ax.annotate(
                    text=f"{float(y):.0f}",
                    xy=(float(x), float(y)),
                    ha=cfg.ANNOTATE_HA,
                    va=cfg.ANNOTATE_VA,
                    fontsize=cfg.TITLE_FONTSIZE,
                    xytext=cfg.ANNOTATE_XYTEXT,
                    textcoords=cfg.ANNOTATE_TEXTCOORDS,
                )

        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()

        return fig, ax
