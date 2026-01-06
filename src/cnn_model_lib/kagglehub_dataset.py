import kagglehub
from pathlib import Path
from .custom_dataset import CustomDataset
from typing import Tuple

def get_bsds500_datasets() -> Tuple[CustomDataset] :

    root = Path(kagglehub.dataset_download(
        "balraj98/berkeley-segmentation-dataset-500-bsds500"
    ))

    if not (root / "images").exists():
        raise FileNotFoundError("The expected 'images' directory was not found in the dataset.")
    
    paths = {
        "train": list((root / "images" / "train").rglob("*.jpg")),
        "test": list((root / "images" / "test").rglob("*.jpg")),
        "val": list((root / "images" / "val").rglob("*.jpg")),
    } 

    return CustomDataset(paths["train"]), CustomDataset(paths["val"]), CustomDataset(paths["test"])


