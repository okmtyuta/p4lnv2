import os

from src.lib.config.dir import Dir
from src.modules.extract.extractor.extractor import Extractor
from src.modules.extract.language.esm.esm2 import ESM2Language
from src.modules.protein.protein_list import ProteinList

# designate the language you want to use for extraction
language = ESM2Language()
# set the language to extractor
extractor = Extractor(language=language)

# read proteins from csv
dataset_csv_path = os.path.join(Dir.root_dir, "data",'busse', "data.csv")
protein_list = ProteinList.from_csv(path=dataset_csv_path)

# execute extraction
extractor(protein_list=protein_list, batch_size=32)

# save extracted features as HDF5 file named `data.h5`
experiment_dir = os.path.join(Dir.root_dir, "result", "EXT0004", "busse")
os.makedirs(experiment_dir, exist_ok=True)
extracted_path = os.path.join(experiment_dir, "data.h5")
protein_list.save_as_hdf5(extracted_path)
