import torch

from src.modules.dataloader.dataloader import DataBatch, Dataloader
from src.modules.model.configurable_model import ConfigurableModel
from src.modules.protein.protein_list import ProteinList
from src.modules.train.criterion import Criterion
from src.modules.train.train_recorder import TrainRecorder
from src.modules.train.types import EpochResult, TrainResult


class Trainer:
    def __init__(self, model: ConfigurableModel, dataloader: Dataloader):
        self._model = model

        self._dataloader = dataloader
        self._train_loader, self._evaluate_loader, self._validate_loader = self._dataloader.rational_split(
            [0.8, 0.1, 0.1]
        )
        self._criterion = Criterion()

        self._model.train()
        self._model.optimizer.train()

        self._recorder = TrainRecorder()

    @property
    def recorder(self):
        return self._recorder

    def _create_epoch_results(self, label: torch.Tensor, output: torch.Tensor, protein_list: ProteinList):
        epoch_results: list[EpochResult] = []
        for i, prop_name in enumerate(self._dataloader.state.output_props):
            _output = output[:, i]
            _label = label[:, i]

            criteria = self._criterion(output=_output, label=_label)

            result: EpochResult = {
                "prop_name": prop_name,
                "epoch": self._recorder.current_epoch,
                "output": _output.tolist(),
                "label": _label.tolist(),
                "criteria": criteria,
            }
            epoch_results.append(result)

        return epoch_results

    def _batch_predict(self, batch: DataBatch, backward: bool = False):
        self._model.optimizer.zero_grad()
        input, label, protein_list = batch.use()

        output = self._model(input=input)
        if backward:
            loss = self._criterion.mean_squared_error(output, label)
            loss.backward()
            self._model.optimizer.step()

        return label, output, protein_list

    def _epoch_predict(self, dataloader: Dataloader, backward: bool = False):
        batch_labels: list[torch.Tensor] = []
        batch_outputs: list[torch.Tensor] = []
        batch_protein_lists: list[ProteinList] = []

        for batch in dataloader.batches:
            _label, _output, _protein_list = self._batch_predict(batch=batch, backward=backward)
            batch_labels.append(_label)
            batch_outputs.append(_output)
            batch_protein_lists.append(_protein_list)

        label = torch.cat(batch_labels)
        output = torch.cat(batch_outputs)
        protein_list = ProteinList.join(batch_protein_lists)

        epoch_results = self._create_epoch_results(label=label, output=output, protein_list=protein_list)
        return epoch_results

    def train(self) -> None:
        while self._recorder.to_continue():
            for i in range(10):
                train_epoch_results = self._epoch_predict(dataloader=self._train_loader, backward=True)
                validate_epoch_results = self._epoch_predict(dataloader=self._validate_loader)
                evaluate_epoch_results = self._epoch_predict(dataloader=self._evaluate_loader)

                self._recorder.append_results(
                    train_epoch_results=train_epoch_results,
                    validate_epoch_results=validate_epoch_results,
                    evaluate_epoch_results=evaluate_epoch_results,
                )

                print(f"Current epoch is {self._recorder._current_epoch}")
                for r in validate_epoch_results:
                    print(f"Validate {r['prop_name']} pearson: {r['criteria']['pearsonr']}")
                for r in evaluate_epoch_results:
                    print(f"Evaluate {r['prop_name']} pearson: {r['criteria']['pearsonr']}")
                for p, r in self._recorder.max_accuracy_result["evaluate"].items():
                    v = self._recorder.max_accuracy_epoch
                    print(f"Max {p} pearson: {r['criteria']['pearsonr']} at {v}")

                self._recorder.next_epoch()

    def as_result(self):
        train_result: TrainResult = {
            "input_props": self._dataloader.state.input_props,
            "output_props": self._dataloader.state.output_props,
            "max_accuracy_epoch": self._recorder.max_accuracy_epoch,
            "max_accuracy_result": self._recorder.max_accuracy_result,
            "train_result": {
                "train": self._recorder.train_result,
                "validate": self._recorder.validate_result,
                "evaluate": self._recorder.evaluate_result,
            },
        }
        return train_result
