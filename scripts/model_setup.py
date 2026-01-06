import cnn_model_lib
from cnn_model_lib import CNNCustomModel, get_bsds500_datasets, calc_confusion_matrix, calc_from_cm
from cnn_model_lib.training import Trainer
import torch
from pathlib import Path
from typing import Optional, List, Dict
import json

import argparse

def training(device: torch.device):
    trainer = Trainer(device=device)
    trainer.train()
    out_path = Path("./model/model.pth").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out_path))

def load_model(device: torch.device) -> CNNCustomModel:
    model_path = Path("./model/model.pth")
    if model_path.exists():
        model = CNNCustomModel().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        raise FileExistsError("Model coundn't be loaded!")
    
def calc_metrics(model: torch.nn.Module, device: torch.device):
    test_dataset = get_bsds500_datasets()[2]
    cm = calc_confusion_matrix(model=model, device=device, ds=test_dataset)

    stats = calc_from_cm(cm)

    return cm, stats

def save_statistics(cm: List[List[int]], statistics: Dict):
    save_path = Path("./results")
    Path.mkdir(save_path, exist_ok=True, parents=True)

    cm_save_path = save_path / "confusion_matrix.json"
    
    with open(cm_save_path, mode="w", encoding="utf-8") as file:
        json.dump(cm, file)
        print("Confusion Matrix saved at: ", cm_save_path.absolute().as_posix())


    stats_save_path = save_path / "summary.json"
    
    with open(stats_save_path, mode="w", encoding="utf-8") as file:
        json.dump(statistics, file)
        print("Statistics saved at: ", stats_save_path.absolute().as_posix())

def init():
    default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parses = argparse.ArgumentParser()
    parses.add_argument("--no-training", dest="training", action="store_false", help="Disable model training")
    parses.set_defaults(training=True, metrics=True)
    
    args = parses.parse_args()

    if args.training:
        training(default_device)
    else:
        print("Skipping model training...")

    model = load_model(default_device)

    cm, statistics = calc_metrics(model, default_device)

    save_statistics(cm.tolist(), statistics)


if __name__ == "__main__":
    init()