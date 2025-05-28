import statistics

from src.modules.protein.protein_list import ProteinProp
from src.modules.train.types import TrainResult


class Analyzer:
    def __init__(self, train_results: list[TrainResult]):
        self._train_results = train_results

    def _get_pearsonrs(self, prop_name: ProteinProp):
        pearsonrs: list[float] = []
        for result in self._train_results:
            pearsonr = result["max_accuracy_result"][prop_name]["pearsonr"]
            pearsonrs.append(pearsonr)

        return pearsonrs

    def _get_rmses(self, prop_name: ProteinProp):
        rmses: list[float] = []
        for result in self._train_results:
            rmse = result["max_accuracy_result"][prop_name]['root_mean_squared_error']
            rmses.append(rmse)

        return rmses

    def get_pearsonr_median(self, prop_name: ProteinProp):
        pearsonrs = self._get_pearsonrs(prop_name)
        median = statistics.median(pearsonrs)

        return median

    def get_pearsonr_mean(self, prop_name: ProteinProp):
        pearsonrs = self._get_pearsonrs(prop_name)
        mean = statistics.mean(pearsonrs)

        return mean

    def get_pearsonr_std(self, prop_name: ProteinProp):
        pearsonrs = self._get_pearsonrs(prop_name)
        std = statistics.stdev(pearsonrs)

        return std
    
    def get_rmse_median(self, prop_name: ProteinProp):
        rmses = self._get_rmses(prop_name)
        median = statistics.median(rmses)

        return median

    def get_rmse_mean(self, prop_name: ProteinProp):
        rmses = self._get_rmses(prop_name)
        mean = statistics.mean(rmses)

        return mean

    def get_rmse_std(self, prop_name: ProteinProp):
        rmses = self._get_rmses(prop_name)
        std = statistics.stdev(rmses)

        return std
