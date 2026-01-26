import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from typing import final, Set, List, Dict, Tuple, cast, Iterable
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

import visualization.config as cfg
from visualization.base_visualizer import BaseVisualizer


@final
class ImageVisualizer(BaseVisualizer):
    def __init__(self, dataset_path: str, img_df_path: str = ""):
        super().__init__(dataset_path, img_df_path)

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

    def compute_image_analysis(self, look_at: str, save_dataframe=False):
        if self.dataframe is None:
            data: List[Dict] = []

            if self.computed_paths is []:
                self.computed_paths = [
                    p
                    for p in self.dataset_path.rglob("*")
                    if p.is_dir()
                    and look_at in str(p)
                    and any(child.is_file() for child in p.iterdir())
                ]

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

            self.dataframe = pd.DataFrame(data)

            if save_dataframe:
                self.dataframe.to_csv("Image_df.csv", index=False)
            else:
                print(self.dataframe)

    def plot_color_channel_distribution(
        self, save_path: str = ""
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot RGB color channel distributions per class using KDE plots.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        channels: List[str] = ["R", "G", "B"]
        classes: List[str] = self.dataframe["Class"].unique().tolist()
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
                subset: pd.DataFrame = self.dataframe[self.dataframe["Class"] == cls]
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

        self._save_figure(fig, save_path)

        return fig, axes

    def plot_mean_rgb_values(self, save_path: str = "") -> Tuple[Figure, Axes]:
        """
        Plot mean RGB values per class as a grouped bar chart.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        mean_vals = self.dataframe.groupby("Class")[["R", "G", "B"]].mean()

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

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

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_image_sharpness(self, save_path: str = "") -> Tuple[Figure, Axes]:
        """
        Plot image sharpness per class using Laplacian variance.
        Log-scaled y-axis is used due to wide range of sharpness values.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

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

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_image_brightness_vs_sharpness(
        self, save_path: str = ""
    ) -> Tuple[Figure, Axes]:
        """
        Scatterplot of Brightness vs Sharpness per class.
        Optionally includes trend lines per class.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()
        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

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

        self._save_figure(fig, save_path)

        return fig, ax

    def plot_file_distribution(self, save_path: str = "") -> Tuple[Figure, Axes]:
        """
        Plot file size distribution per class using a log-scaled boxplot.
        Median values are annotated for clarity.
        """

        if self.dataframe is None:
            raise RuntimeError(
                "DataFrame is not loaded. Call compute_image_analysis() first."
            )

        df_copy: pd.DataFrame = self.dataframe.copy()

        # Avoid zero values for log scale
        df_copy["FileSizeKB_safe"] = df_copy["FileSizeKB"].clip(lower=cfg.FILESIZE)

        fig: Figure
        ax: Axes
        fig, ax = self._create_fig_ax()

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

        self._save_figure(fig, save_path)

        return fig, ax
