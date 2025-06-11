import os

from src.lib.config.dir import Dir
from src.modules.train.train_result import TrainResultLoader
from src.modules.visualize.visualizer import Visualizer

train_result_loader = TrainResultLoader.from_h5(path=os.path.join(Dir.result_dir, "TRN0001", "0.h5"))
visualizer = Visualizer(train_result=train_result_loader.train_result)
visualizer.save_learning_result(path="test.png", prop_name="rt")
visualizer.save_evaluate_max_accuracy_scatter(path="test_sc.png", prop_name="rt")
