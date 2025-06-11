import os

from src.lib.config.dir import Dir
from src.modules.extract.extractor.extractor import Extractor
from src.modules.extract.language.quick_esm.quick_esm2 import QuickESM2Language
from src.modules.protein.protein_list import ProteinList

deeplc_dataset_names = ["scx"]

# designate the language you want to use for extraction
language = QuickESM2Language()
# set the language to extractor
extractor = Extractor(language=language)

for i, dataset_name in enumerate(deeplc_dataset_names):
    # read proteins from csv
    dataset_csv_path = os.path.join(Dir.root_dir, "data", dataset_name, "data.csv")
    protein_list = ProteinList.from_csv(path=dataset_csv_path)

    # execute extraction
    extractor(protein_list=protein_list, batch_size=32)

    # save extracted features as HDF5 file named `data.h5`
    experiment_dir = os.path.join(Dir.root_dir, "result", "EXT0001", dataset_name)
    os.makedirs(experiment_dir, exist_ok=True)
    extracted_path = os.path.join(experiment_dir, "quick-data.h5")
    protein_list.save_as_hdf5(extracted_path)
