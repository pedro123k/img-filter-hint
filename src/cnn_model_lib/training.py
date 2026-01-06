from . import cnn_model
from . import kagglehub_dataset as syn_dataset
from typing import Dict, Optional
from torch.utils.data import DataLoader
import torch
from torch import nn as nn
import tqdm

class Trainer:
    class EarlyStopping:
        def __init__(self, patience=3, mode="min"):
            self._patience = patience
            self._count = 0
            self._best_metric = None
            self._total_calls = 0
            self._mode = mode

        def _is_improvement(self, metric):
            if self._best_metric is None:
                return True

            if self._mode == "min":
                return metric < self._best_metric
            else:
                return metric > self._best_metric

        def __call__(self, metric) -> bool:
            self._total_calls += 1

            if self._is_improvement(metric):
                self._best_metric = metric
                self._count = 0
            else:
                self._count += 1

            return self._count >= self._patience

        @property
        def total(self):
            return self._total_calls



    def __init__(self, device=torch.device, syn_params: Optional[Dict] = None):
        self._syn_params = syn_params
        self._device = device
        self._model = cnn_model.CNNCustomModel().to(self._device)

        datasets = syn_dataset.get_bsds500_datasets()

        self._train_ds = datasets[0]
        self._val_ds = datasets[1]
        self._test_ds = datasets[2]

    def train(self) -> None:
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        loss = torch.nn.CrossEntropyLoss()

        early_stopper = Trainer.EarlyStopping(patience=5)

        num_epochs = 20

        pbar = tqdm.tqdm(total=num_epochs, desc="Training Progress: ")

        dataloader_train = DataLoader(self._train_ds, batch_size=32, shuffle=True)
        dataloader_val = DataLoader(self._val_ds, batch_size=32)
        dataloader_test = DataLoader(self._test_ds, batch_size=32)

        for epoch in range(num_epochs):
            
            agg_loss = 0.0

            for x, y in dataloader_train:
                self._model.train()

                x_train, y_train = x.to(self._device), y.to(self._device)

                optimizer.zero_grad()
                outs = self._model(x_train)
                l = loss(outs, y_train)
                l.backward()
                optimizer.step()
                agg_loss += l.detach().cpu().item()

            self._model.eval()
            agg_loss_val = 0.0

            with torch.no_grad():
                for x, y in dataloader_val:
                    self._model.train()

                    x_val, y_val = x.to(self._device), y.to(self._device)
                    outs = self._model(x_val)
                    l = loss(outs, y_val)
                    agg_loss_val += l.item()

            if early_stopper(agg_loss_val):
                pbar.close()
                print(f"Early Stopping at epoch = {early_stopper.total} / {num_epochs} ")
                break

            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            pbar.set_postfix(loss=agg_loss, val_loss=agg_loss_val)
            pbar.update(1)

        pbar.close()
        
        total = 0.0
        corrects = 0.0

        self._model.eval()

        with torch.no_grad():
            for x, y in dataloader_test:
                x_test, y_test = x.to(self._device), y.to(self._device)
                outs = torch.argmax(self._model(x_test), dim=1)

                corrects += (y_test == outs).long().sum().item()
                total += y_test.numel()

        acc = 100 * corrects / total 

        print(f"Metric-Eval(Acc): {acc}")
        print("Training is done!")

    @property
    def model(self) -> cnn_model.CNNCustomModel:
        return self._model
        

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)