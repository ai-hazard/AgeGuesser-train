
from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
from torch import nn
from models.torch.ageguesser import AgeNetwork

def get_bin(age):
    if age <= 10:
      return 0
    elif age > 10 and age <= 20:
      return 1
    elif age > 20 and age <= 30:
      return 2
    elif age > 30 and age <= 40:
      return 3
    elif age > 40 and age <= 50:
      return 4
    elif age > 50 and age <= 60:
      return 5
    elif age > 60 and age <= 70:
      return 6
    elif age > 70:
      return 7

def update_errors_dict(errors_loss, y_true, y_pred):
  
    y_true_ = y_true.detach().cpu().numpy().reshape(-1)
    y_pred_ = y_pred.detach().cpu().numpy().reshape(-1)

    for i in range(len(y_true_)):
        e = np.abs(y_true_[i] - y_pred_[i])
        errors_loss[get_bin(y_true_[i])]["err"] += e
        errors_loss[get_bin(y_true_[i])]["num"] += 1
    return errors_loss

def compute_reg32(errors_loss, mae):
    n = 8
    for a in errors_loss:
        if errors_loss[a]["num"] == 0:
            n -= 1
            continue
        errors_loss[a]["mae"] = errors_loss[a]["err"] / errors_loss[a]["num"]
    regularity = 0

    sum_diff = 0
    for a in errors_loss:
        if errors_loss[a]["num"] == 0:
            continue
        sum_diff += pow((errors_loss[a]["mae"] - mae), 2)
    #print(errors)
    if n > 0:
        regularity = sum_diff / 8
    else:
        regularity = 0
    return np.sqrt(regularity)

class AgeNetworkPL(pl.LightningModule):
    def __init__(self, model: AgeNetwork, lr: float = 1e-3, ro: float = 1.0, phi: float = 1.25):
        super().__init__()
        
        self.lr = lr
        self.ro = ro
        self.phi = phi
        self.save_hyperparameters(ignore=["model"])
        
        self.model = model
        # loss
        self.loss_fn = nn.L1Loss()
        self.loss_fn_nored = nn.L1Loss(reduction='none')

        # logging
        self.stats = {}
        self.errors_loss = {}
        self.i = 0
        for i in range(8):
            self.errors_loss[i] = { "err": 0, "num":0, "mae":0 }
        self.mae_per_class = {}

        self.sums = {"loss_mae": 0, "loss_reg": 0, "loss_tot":0, "loss_mae_w": 0,}
        self.avgs = {"loss_mae": 0, "loss_reg": 0, "loss_tot":0, "loss_mae_w": 0,}
    
    def forward(self, x):
        x = self.model(x)
        return x


    def update_stats(self, loss_mae_clean, loss_weighted_value, reg, tot_loss):
        self.sums["loss_mae"] += loss_mae_clean
        self.sums["loss_mae_w"] += loss_weighted_value
        self.sums["loss_reg"] += reg
        self.sums["loss_tot"] += tot_loss

        self.avgs["loss_mae"] = (self.sums["loss_mae"]/self.i)
        self.avgs["loss_mae_w"] = (self.sums["loss_mae_w"]/self.i)
        self.avgs["loss_reg"] = (self.sums["loss_reg"]/self.i)
        self.avgs["loss_tot"] = (self.sums["loss_tot"]/self.i)

    def loss(self, pred, y):
        loss_mae_ = self.loss_fn_nored(pred, y,)

        loss_mae_clean = float(loss_mae_.mean().item()) # just MAE

        loss_mae_weighted = torch.where(y>=60, (loss_mae_ * self.phi), loss_mae_) # MAE with critical classes penalty

        loss_mae = loss_mae_weighted.mean()

        loss_weighted_value = float(loss_mae.item())
        
        # update regularity
        self.errors_loss = update_errors_dict(self.errors_loss, y, pred)
        reg = compute_reg32(self.errors_loss, loss_mae_clean)

        final_loss = loss_mae + (reg * self.ro)

        return final_loss, loss_weighted_value, loss_mae_clean, reg


    def training_step(self, batch):
        
        img, target = batch
        x = self.forward(img)
        
        self.i += 1

        final_loss, loss_weighted_value, loss_mae_clean, reg = self.loss(x, target)
        
        final_loss_value = final_loss.item()
        self.update_stats(loss_mae_clean, loss_weighted_value, reg, final_loss_value )
        
        self.log(
            "train/mae",
            value=loss_mae_clean,
            on_step=True,
            prog_bar=True
        )
        self.log(
            "train/regularity",
            value=reg,
            on_step=True,
            prog_bar=True
        )

        self.log(
            "train/w_mae",
            value=final_loss_value,
            on_step=True,
            prog_bar=True
        )

        return {"loss": final_loss}

    def on_train_start(self) -> None:
        super().on_train_start()
        self.sums = {"loss_mae": 0, "loss_reg": 0, "loss_tot":0, "loss_mae_w": 0,}
        self.avgs = {"loss_mae": 0, "loss_reg": 0, "loss_tot":0, "loss_mae_w": 0,}

    def on_train_epoch_end(self) -> None:
       super().on_train_epoch_end()
       self.i = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
