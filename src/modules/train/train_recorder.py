from typing import Optional

from src.lib.config.Env import Env
from src.modules.train.types import (
    EpochResult,
    TrainRecorderMaxAccuracyResult,
    TrainRecorderResult,
    TrainRecorderResultKey,
)


class TrainRecorder:
    def __init__(self):
        self._current_epoch = 1
        self._max_accuracy_epoch: Optional[int] = None

        self._result: TrainRecorderResult = {
            "train": {},
            "evaluate": {},
            "validate": {},
        }
        self._max_accuracy_result: TrainRecorderMaxAccuracyResult = {
            "train": {},
            "evaluate": {},
            "validate": {},
        }

    @property
    def train_result(self):
        return self._result["train"]

    @property
    def validate_result(self):
        return self._result["validate"]

    @property
    def evaluate_result(self):
        return self._result["evaluate"]

    @property
    def current_epoch(self):
        return self._current_epoch

    @property
    def max_accuracy_result(self):
        return self._max_accuracy_result

    @property
    def max_accuracy_epoch(self):
        return self._max_accuracy_epoch

    def _set_max_accuracy_result(self, key: TrainRecorderResultKey, epoch_results: list[EpochResult]):
        for result in epoch_results:
            prop_name = result["prop_name"]
            self._max_accuracy_result[key][prop_name] = result

    def _append_result(self, key: TrainRecorderResultKey, epoch_results: list[EpochResult]):
        for result in epoch_results:
            prop_name = result["prop_name"]
            results = self._result[key].get(prop_name)
            if results is None:
                self._result[key][prop_name] = [result]
            else:
                self._result[key][prop_name].append(result)

    def is_max_accuracy(self, epoch_results: list[EpochResult]):
        if self._max_accuracy_result is None:
            return True

        pearsonrs = map(lambda result: result["criteria"]["pearsonr"], epoch_results)
        accuracy = sum(pearsonrs)

        max_accuracy_validate_result = self._max_accuracy_result["validate"]
        max_accuracy_pearsonrs = map(
            lambda result: result["criteria"]["pearsonr"], max_accuracy_validate_result.values()
        )
        max_accuracy = sum(max_accuracy_pearsonrs)

        return accuracy > max_accuracy

    def next_epoch(self):
        self._current_epoch += 1

    def append_results(
        self,
        train_epoch_results: list[EpochResult],
        validate_epoch_results: list[EpochResult],
        evaluate_epoch_results: list[EpochResult],
    ):
        self._append_result(key="train", epoch_results=train_epoch_results)
        self._append_result(key="validate", epoch_results=validate_epoch_results)
        self._append_result(key="evaluate", epoch_results=evaluate_epoch_results)

        if self.is_max_accuracy(epoch_results=validate_epoch_results):
            self._set_max_accuracy_result(key="train", epoch_results=train_epoch_results)
            self._set_max_accuracy_result(key="validate", epoch_results=validate_epoch_results)
            self._set_max_accuracy_result(key="evaluate", epoch_results=evaluate_epoch_results)
            self._max_accuracy_epoch = self._current_epoch

    def to_continue(self):
        if self._max_accuracy_epoch is None:
            return True

        return self._current_epoch - self._max_accuracy_epoch < Env.continuous_epochs
