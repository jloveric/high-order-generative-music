from typing import List
from torch import Tensor
from omegaconf import DictConfig
from torchmetrics.functional import accuracy
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import (
    HighOrderFullyConvolutionalNetwork,
    HighOrderMLP,
)
from pytorch_lightning import LightningModule
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch

import torch_optimizer as alt_optim

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.conv = HighOrderFullyConvolutionalNetwork(
            layer_type=cfg.conv.layer_type,
            n=cfg.conv.n,
            channels=cfg.conv.channels,
            segments=cfg.conv.segments,
            kernel_size=cfg.conv.kernel_size,
            rescale_output=False,
            periodicity=cfg.conv.periodicity,
            normalization=torch.nn.LazyBatchNorm1d,
            stride=cfg.conv.stride,
            pooling=None,  # don't add an average pooling layer
        )
        """
        self.mlp = HighOrderMLP(
            layer_type=cfg.mlp.layer_type,
            n=cfg.mlp.n,
            n_in=cfg.mlp.n_in,
            n_hidden=cfg.mlp.n_in,
            n_out=cfg.mlp.n_out,
            in_width=cfg.mlp.input.width,
            in_segments=cfg.mlp.input.segments,
            out_width=cfg.mlp.output.width,
            out_segments=cfg.mlp.output.segments,
            hidden_width=cfg.mlp.hidden.width,
            hidden_layers=cfg.mlp.hidden.layers,
            hidden_segments=cfg.mlp.hidden.segments,
            normalization=torch.nn.LazyBatchNorm1d,
        )
        """
        self.linear = torch.nn.LazyLinear(out_features=1)

        self.loss = nn.MSELoss()
        self.model = nn.Sequential([self.conv, self.linear])

    def forward(self, x):
        return self.model(x)

    def eval_step(self, batch: Tensor, name: str):
        x, y = batch
        y_hat = self(x.flatten(1))
        loss = self.loss(y_hat.flatten(), y.flatten())

        self.log(f"{name}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def configure_optimizers(self):
        if self.cfg.optimizer.name == "adahessian":
            return alt_optim.Adahessian(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                betas=self.cfg.optimizer.betas,
                eps=self.cfg.optimizer.eps,
                weight_decay=self.cfg.optimizer.weight_decay,
                hessian_power=self.cfg.optimizer.hessian_power,
            )
        elif self.cfg.optimizer.name == "adam":

            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.cfg.optimizer.lr,
            )

            reduce_on_plateau = False
            if self.cfg.optimizer.scheduler == "plateau":
                logger.info("Reducing lr on plateau")
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.cfg.optimizer.patience,
                    factor=self.cfg.optimizer.factor,
                    verbose=True,
                )
                reduce_on_plateau = True
            elif self.cfg.optimizer.scheduler == "exponential":
                logger.info("Reducing lr exponentially")
                lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.cfg.optimizer.gamma
                )
            else:
                return optimizer

            scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": reduce_on_plateau,
                "monitor": "train_loss",
            }
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Optimizer {self.cfg.optimizer.name} not recognized")
