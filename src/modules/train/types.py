from typing import Literal, TypedDict

from src.modules.protein.protein import ProteinProps
from src.modules.protein.protein_list import ProteinProp

TrainRecorderResultKey = Literal["train", "evaluate", "validate"]


class Criteria(TypedDict):
    root_mean_squared_error: float
    mean_squared_error: float
    mean_absolute_error: float
    pearsonr: float


class EpochResult(TypedDict):
    prop_name: ProteinProp
    epoch: int
    label: list[float]
    output: list[float]
    criteria: Criteria


class TrainResult(TypedDict):
    input_props: list[ProteinProps]
    output_props: list[ProteinProps]
    max_accuracy_epoch: list
    max_accuracy_result: dict[TrainRecorderResultKey, dict[ProteinProp, EpochResult]]
    train_result: dict[TrainRecorderResultKey, dict[ProteinProp, list[EpochResult]]]


TrainRecorderResult = dict[TrainRecorderResultKey, dict[ProteinProp, list[EpochResult]]]
TrainRecorderMaxAccuracyResult = dict[TrainRecorderResultKey, dict[ProteinProp, EpochResult]]
