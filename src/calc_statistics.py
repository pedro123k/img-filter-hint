import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def calc_confusion_matrix(model: torch.nn.Module, device: torch.device , ds: Dataset) -> np.ndarray:

    cm = np.zeros((4,4), dtype=np.int64)

    dataloader = DataLoader(ds, batch_size=32)

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outs = torch.argmax(model(x), dim=1)
            
            for real, pred in zip(y.cpu().numpy(), outs.cpu().numpy()): 
                cm[real][pred] += 1

    return cm

def calc_from_cm(cm: np.ndarray) -> dict:
    statistics = {
        "OK": {},
        "NOISY": {},
        "BLURRED": {},
        "LOW_LIGHTED": {}
    }

    eps = 1e-8

    acc = np.sum(cm[(0, 1, 2, 3), (0, 1, 2, 3)]) / np.sum(cm)

    statistics["acc"] = acc.item()

    tp = lambda i: cm[i,i]
    fp = lambda i: np.sum(cm[:, i]) - cm[i,i]
    fn = lambda i: np.sum(cm[i, :]) - cm[i,i]
    precision = lambda tp, fp: (tp / (tp + fp + eps)).item()
    recall = lambda tp, fn: (tp / (tp + fn + eps)).item()
    f1_score = lambda precision, recall: 2*(precision * recall) / (precision + recall + eps)

    f1_geral = 0;0

    for i, cat in enumerate(["OK", "NOISY", "BLURRED", "LOW_LIGHTED"]):
        cat_precision = precision(tp(i), fp(i))
        cat_recall = recall(tp(i), fn(i))
        cat_f1score = f1_score(cat_precision, cat_recall)
        f1_geral += cat_f1score

        statistics[cat]["precision"] = cat_precision
        statistics[cat]["recall"] = cat_recall
        statistics[cat]["f1_score"] = cat_f1score
    
    f1_geral /= 4

    statistics["f1_global"] = f1_geral 

    return statistics




