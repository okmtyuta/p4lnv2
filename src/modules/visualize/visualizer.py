import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes

from src.modules.color.ColorPallet import ColorPallet
from src.modules.protein.protein_list import ProteinProp
from src.modules.train.types import TrainRecorderResultKey, TrainResult


class Visualizer:
    @classmethod
    def save_histogram(cls, path: str, prop_name: ProteinProp):
        df = pl.read_csv(path)
        df = df.filter(pl.col("ccs").is_null())
        # plt.hist(values, bins=50, color=ColorPallet.hex_universal_color["blue"])
        # plt.axvline(
        #     mean,
        #     color=ColorPallet.hex_universal_color["red"],
        #     linestyle="dashed",
        #     linewidth=2,
        #     label=f"平均: {mean:.2f}",
        # )

        # plt.show()

    def __init__(self, train_result: TrainResult):
        self._train_result = train_result
        self._pallet = ColorPallet()

    def save_learning_result(self, path: str, prop_name: ProteinProp):
        figure = plt.figure(dpi=100, figsize=(18, 12))
        figure.subplots_adjust(left=0.1, right=0.75, bottom=0.2, top=0.85)
        left_axes = figure.add_subplot(1, 1, 1)
        self._render_loss_curve(key="train", axes=left_axes, prop_name=prop_name)
        self._render_loss_curve(key="validate", axes=left_axes, prop_name=prop_name)
        self._render_loss_curve(key="evaluate", axes=left_axes, prop_name=prop_name)

        right_axes = left_axes.twinx()
        self._render_pearsonr_curve(key="train", axes=right_axes, prop_name=prop_name)
        self._render_pearsonr_curve(key="validate", axes=right_axes, prop_name=prop_name)
        self._render_pearsonr_curve(key="evaluate", axes=right_axes, prop_name=prop_name)

        self._render_evaluate_max_accuracy_pearsonr(axes=right_axes, prop_name=prop_name)

        left_axes_handles, left_axes_labels = left_axes.get_legend_handles_labels()
        right_axes_handles, right_axes_labels = right_axes.get_legend_handles_labels()
        left_axes.legend(
            left_axes_handles + right_axes_handles,
            left_axes_labels + right_axes_labels,
            bbox_to_anchor=(1.1, 1),
            loc="upper left",
            borderaxespad=0,
        )
        plt.savefig(path)
        plt.close()

    def _render_loss_curve(self, key: TrainRecorderResultKey, axes: Axes, prop_name: ProteinProp):
        results = self._train_result["train_result"][key][prop_name]

        epochs = [result["epoch"] for result in results]
        root_mean_squared_errors = [result["criteria"]["root_mean_squared_error"] for result in results]

        color = self._pallet.consume_current_color()
        axes.plot(epochs, root_mean_squared_errors, color=color, label=f"{key} Loss")

    def _render_pearsonr_curve(self, key: TrainRecorderResultKey, axes: Axes, prop_name: ProteinProp):
        results = self._train_result["train_result"][key][prop_name]

        epochs = [result["epoch"] for result in results]
        pearsonrs = [result["criteria"]["pearsonr"] for result in results]

        color = self._pallet.consume_current_color()
        axes.plot(epochs, pearsonrs, color=color, label=f"{key} Pearson")

    def _render_evaluate_max_accuracy_pearsonr(self, axes: Axes, prop_name: ProteinProp):
        result = self._train_result["max_accuracy_result"]["evaluate"][prop_name]

        pearsonr = result["criteria"]["pearsonr"]
        epoch = result["epoch"]
        color = self._pallet.consume_current_color()

        axes.axhline(y=pearsonr, xmin=0, xmax=1, color=color, linestyle="--")
        axes.axvline(x=epoch, ymin=0, ymax=1, color=color, linestyle="--")

    def save_evaluate_max_accuracy_scatter(self, path: str, prop_name: ProteinProp):
        figure = plt.figure(dpi=100, figsize=(8, 6))
        # figure.subplots_adjust(left=0.1, right=0.75, bottom=0.2, top=0.85)
        axes = figure.add_subplot(1, 1, 1)

        label = self._train_result["max_accuracy_result"]["evaluate"][prop_name]["label"]
        output = self._train_result["max_accuracy_result"]["evaluate"][prop_name]["output"]

        xy_min = min(label + output)
        xy_max = max(label + output)

        axes.scatter(
            label,
            output,
            color=ColorPallet.hex_universal_color["red"],
            s=2,
        )
        axes.plot([xy_min, xy_max], [xy_min, xy_max], color=ColorPallet.hex_universal_color["blue"])
        axes.set_xlabel(f"Observed {prop_name} value")
        axes.set_ylabel(f"Predicted {prop_name} value")
        plt.savefig(path)
        plt.close()
