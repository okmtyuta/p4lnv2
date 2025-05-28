from typing import Literal, Optional

from src.modules.protein.protein_list import ProteinProp
from src.modules.train.types import EpochResult

TrainRecorderResultKind = Literal["train", "evaluate", "validate"]
TrainRecorderResult = dict[TrainRecorderResultKind, dict[ProteinProp, list[EpochResult]]]


class TrainRecorder:
    def __init__(self):
        self._current_epoch = 1
        self._max_accuracy_result: Optional[dict[ProteinProp, EpochResult]] = None
        self._max_accuracy_epoch: Optional[int] = None

        self._result: TrainRecorderResult = {
            "train": {},
            "evaluate": {},
            "validate": {},
        }

        self._train_result: dict[ProteinProp, list[EpochResult]] = {}
        self._evaluate_result: dict[ProteinProp, list[EpochResult]] = {}
        self._validate_result: dict[ProteinProp, list[EpochResult]] = {}

    @property
    def current_epoch(self):
        return self._current_epoch

    @property
    def max_accuracy_result(self):
        if self._max_accuracy_result is None:
            raise Exception
        return self._max_accuracy_result

    @property
    def max_accuracy_epoch(self):
        return self._max_accuracy_epoch

    @property
    def train_result(self):
        return self._train_result

    @property
    def evaluate_result(self):
        return self._evaluate_result

    def is_max_accuracy(self, epoch_results: list[EpochResult]):
        if self._max_accuracy_result is None:
            return True

        pearsonrs = map(lambda result: result["criteria"]["pearsonr"], epoch_results)
        accuracy = sum(pearsonrs)

        max_accuracy_pearsonrs = map(lambda result: result["criteria"]["pearsonr"], self._max_accuracy_result.values())
        max_accuracy = sum(max_accuracy_pearsonrs)

        return accuracy > max_accuracy

    def next_epoch(self):
        self._current_epoch += 1

    def append_result(self, kind: TrainRecorderResultKind, epoch_results: list[EpochResult]):
        for result in epoch_results:
            prop_name = result["prop_name"]
            results = self._train_result.get(prop_name)
            if results is None:
                self._train_result[prop_name] = [result]
            else:
                self._train_result[prop_name].append(result)

        if kind == "validate" and self.is_max_accuracy(epoch_results=epoch_results):
            max_accuracy_result = {}
            for result in epoch_results:
                max_accuracy_result[result["prop_name"]] = result
            self._max_accuracy_result = max_accuracy_result
            self._max_accuracy_epoch = self._current_epoch

    def to_continue(self):
        if self._max_accuracy_epoch is None:
            return True

        return self._current_epoch - self._max_accuracy_epoch < 300
