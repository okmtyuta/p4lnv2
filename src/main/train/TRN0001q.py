import os

from src.lib.config.dir import Dir
from src.modules.data_pipeline.aggregator import Aggregator
from src.modules.data_pipeline.data_pipeline import DataPipeline
from src.modules.data_pipeline.initializer import Initializer
from src.modules.data_pipeline.sinusoidal_positional_encoder import (
    SinusoidalPositionalEncoder,
    BidirectionalSinusoidalPositionalEncoder,
)
from src.modules.dataloader.dataloader import Dataloader, DataloaderState
from src.modules.model.architecture import Architecture
from src.modules.model.configurable_model import ConfigurableModel
from src.modules.protein.protein_list import ProteinList
from src.modules.train.train_result import TrainResultLoader
from src.modules.train.trainer import Trainer

protein_list = ProteinList.from_hdf5("result/EXT0001q/plasma_lumos_1h/data.h5")

aggregator = Aggregator("mean")
initializer = Initializer()
positional_encoder = SinusoidalPositionalEncoder(a=1000, b=1, gamma=0)
pipeline = DataPipeline(pipes=[initializer, positional_encoder, aggregator])
dataloader_state = DataloaderState(
    {
        "protein_list": protein_list,
        "batch_size": 128,
        "input_props": ["length"],
        "output_props": ["rt"],
        "pipeline": pipeline,
        "cacheable": True,
    }
)
dataloader = Dataloader(state=dataloader_state)


for i in range(1):
    architecture = Architecture(source=(128, 5), input_size=1280 + 1, output_size=1)
    model = ConfigurableModel(architecture=architecture)
    trainer = Trainer(model=model, dataloader=dataloader)
    trainer.train()

    # train_result = trainer.as_result()
    # train_result_loader = TrainResultLoader(train_result)
    # save_dir = os.path.join(Dir.result_dir, "TRN0001")
    # os.makedirs(save_dir, exist_ok=True)

    # train_result_loader.save_as_h5(os.path.join(save_dir, f"{i}.h5"))
