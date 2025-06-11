import h5py

from src.modules.train.trainer import Trainer
from src.modules.train.types import EpochResult, TrainResult


class TrainResultLoader:
    def __init__(self, train_result: TrainResult):
        self._train_result = train_result

    @property
    def train_result(self):
        return self._train_result

    def _set_epoch_result(cls, group: h5py.Group, epoch_result: EpochResult):
        group.attrs["prop_name"] = epoch_result["prop_name"]
        group.attrs["epoch"] = epoch_result["epoch"]

        group.create_dataset("label", data=epoch_result["label"])
        group.create_dataset("output", data=epoch_result["output"])

        criteria_group = group.create_group("criteria")
        criteria_group.create_dataset("pearsonr", data=epoch_result["criteria"]["pearsonr"])
        criteria_group.create_dataset("mean_squared_error", data=epoch_result["criteria"]["mean_squared_error"])
        criteria_group.create_dataset(
            "root_mean_squared_error", data=epoch_result["criteria"]["root_mean_squared_error"]
        )
        criteria_group.create_dataset("mean_absolute_error", data=epoch_result["criteria"]["mean_absolute_error"])

    def save_as_h5(self, path: str):
        with h5py.File(path, mode="w") as f:
            result_group = f.create_group("result")
            result_group.attrs["input_props"] = self._train_result["input_props"]
            result_group.attrs["output_props"] = self._train_result["output_props"]
            result_group.attrs["max_accuracy_epoch"] = self._train_result["max_accuracy_epoch"]

            for prop_name in self._train_result["output_props"]:
                prop_group = result_group.create_group(prop_name)
                max_accuracy_group = prop_group.create_group("max_accuracy")

                train_max_accuracy_group = max_accuracy_group.create_group("train")
                validate_max_accuracy_group = max_accuracy_group.create_group("validate")
                evaluate_max_accuracy_group = max_accuracy_group.create_group("evaluate")
                self._set_epoch_result(
                    train_max_accuracy_group, self._train_result["max_accuracy_result"]["train"][prop_name]
                )
                self._set_epoch_result(
                    validate_max_accuracy_group, self._train_result["max_accuracy_result"]["validate"][prop_name]
                )
                self._set_epoch_result(
                    evaluate_max_accuracy_group, self._train_result["max_accuracy_result"]["evaluate"][prop_name]
                )

                train_group = prop_group.create_group("train")
                train_epochs_group = train_group.create_group("epochs")
                train_epoch_results = self._train_result["train_result"]["train"][prop_name]
                for epoch_result in train_epoch_results:
                    epoch_group = train_epochs_group.create_group(str(epoch_result["epoch"]))
                    self._set_epoch_result(epoch_group, epoch_result)

                evaluate_group = prop_group.create_group("evaluate")
                evaluate_epochs_group = evaluate_group.create_group("epochs")
                evaluate_epoch_results = self._train_result["train_result"]["evaluate"][prop_name]
                for epoch_result in evaluate_epoch_results:
                    epoch_group = evaluate_epochs_group.create_group(str(epoch_result["epoch"]))
                    self._set_epoch_result(epoch_group, epoch_result)

                validate_group = prop_group.create_group("validate")
                validate_epochs_group = validate_group.create_group("epochs")
                validate_epoch_results = self._train_result["train_result"]["validate"][prop_name]
                for epoch_result in validate_epoch_results:
                    epoch_group = validate_epochs_group.create_group(str(epoch_result["epoch"]))
                    self._set_epoch_result(epoch_group, epoch_result)

    @classmethod
    def _load_epoch_result(self, group: h5py.Group):
        prop_name = group.attrs["prop_name"]
        epoch = group.attrs["epoch"].item()

        label = group["label"][:].tolist()
        output = group["output"][:].tolist()

        criteria_group = group["criteria"]
        pearsonr = criteria_group["pearsonr"][()].item()
        mean_squared_error = criteria_group["mean_squared_error"][()].item()
        root_mean_squared_error = criteria_group["root_mean_squared_error"][()].item()
        mean_absolute_error = criteria_group["mean_absolute_error"][()].item()

        epoch_result: EpochResult = {
            "prop_name": prop_name,
            "epoch": epoch,
            "label": label,
            "output": output,
            "criteria": {
                "pearsonr": pearsonr,
                "mean_squared_error": mean_squared_error,
                "root_mean_squared_error": root_mean_squared_error,
                "mean_absolute_error": mean_absolute_error,
            },
        }
        return epoch_result

    @classmethod
    def from_trainer(cls, trainer: Trainer):
        train_result = trainer.as_result()
        return TrainResultLoader(train_result=train_result)

    @classmethod
    def from_h5(cls, path: str):
        print(f"loading {path}...")
        with h5py.File(path, mode="r") as f:
            result_group = f["result"]

            input_props = result_group.attrs["input_props"]
            output_props = result_group.attrs["output_props"]
            max_accuracy_epoch = result_group.attrs["max_accuracy_epoch"]

            max_accuracy_result = {"train": {}, "validate": {}, "evaluate": {}}
            _train_result = {}
            _validate_result = {}
            _evaluate_result = {}

            for prop_name in output_props:
                prop_group = result_group[prop_name]
                max_accuracy_group = prop_group["max_accuracy"]

                max_accuracy_result["train"][prop_name] = cls._load_epoch_result(max_accuracy_group["train"])
                max_accuracy_result["validate"][prop_name] = cls._load_epoch_result(max_accuracy_group["validate"])
                max_accuracy_result["evaluate"][prop_name] = cls._load_epoch_result(max_accuracy_group["evaluate"])

                train_group = prop_group["train"]
                train_epochs_group = train_group["epochs"]
                prop_train_results: list[EpochResult] = []
                for epoch in train_epochs_group.keys():
                    epoch_group = train_epochs_group[epoch]
                    train_epoch_result = cls._load_epoch_result(epoch_group)
                    prop_train_results.append(train_epoch_result)

                prop_train_results = sorted(prop_train_results, key=lambda result: result["epoch"])
                _train_result[prop_name] = prop_train_results

                evaluate_group = prop_group["evaluate"]
                evaluate_epochs_group = evaluate_group["epochs"]
                prop_evaluate_results: list[EpochResult] = []
                for epoch in evaluate_epochs_group.keys():
                    epoch_group = evaluate_epochs_group[epoch]
                    evaluate_epoch_result = cls._load_epoch_result(epoch_group)
                    prop_evaluate_results.append(evaluate_epoch_result)

                prop_evaluate_results = sorted(prop_evaluate_results, key=lambda result: result["epoch"])
                _evaluate_result[prop_name] = prop_evaluate_results

                validate_group = prop_group["validate"]
                validate_epochs_group = validate_group["epochs"]
                prop_validate_results: list[EpochResult] = []
                for epoch in validate_epochs_group.keys():
                    epoch_group = validate_epochs_group[epoch]
                    validate_epoch_result = cls._load_epoch_result(epoch_group)
                    prop_validate_results.append(validate_epoch_result)

                prop_validate_results = sorted(prop_validate_results, key=lambda result: result["epoch"])
                _validate_result[prop_name] = prop_validate_results

        train_result: TrainResult = {
            "input_props": input_props,
            "output_props": output_props,
            "max_accuracy_epoch": max_accuracy_epoch,
            "max_accuracy_result": max_accuracy_result,
            "train_result": {"train": _train_result, "validate": _validate_result, "evaluate": _evaluate_result},
        }

        return TrainResultLoader(train_result=train_result)
