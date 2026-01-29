from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import src.visualization.config as cfg


class BaseVisualizer:
    def __init__(self, dataset_path: str, dataframe_path: str = ""):
        self.dataset_path: Path = Path(dataset_path)
        self.dataframe_path: Path = Path(dataframe_path)

        self.dataframe: Optional[pd.DataFrame] = None

        self._read_files()

    def _read_files(self) -> None:
        if self.dataframe_path.name != "" and self.dataframe_path.exists():
            self.dataframe = pd.read_csv(self.dataframe_path)
            if "Path" in self.dataframe.columns:
                self.dataframe = self.dataframe.set_index("Path")

    def _create_fig_ax(self) -> Tuple[Figure, Axes]:
        return plt.subplots(
            figsize=cfg.FIGSIZE_MEDIUM, constrained_layout=cfg.CONSTRAINED_LAYOUT
        )

    def _save_figure(self, fig: Figure, save_path: str = ""):
        if save_path:
            fig.savefig(save_path, dpi=cfg.DPI, bbox_inches=cfg.BBOX_INCHES)
            plt.close(fig)
        else:
            plt.show()
