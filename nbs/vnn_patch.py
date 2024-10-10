#def train_one_model():
#    print('No one expects the spanish inquisition!')

import torch
import torch.nn.functional as F # F.mse_loss
import lightning as pl


class plDNN_general(pl.LightningModule):
    def __init__(self, mod, log_weight_stats = False):
        super().__init__()
        self.mod = mod
        self.log_weight_stats = log_weight_stats
        # for demonstration purposes only
        self.bar = '|'+''.join(['-' for i in range(99)])
        
    def training_step(self, batch, batch_idx):
        # for demonstration purposes only
        print(self.bar)
        self.bar = self.bar[-1]+self.bar[:-1]

        y_i, x_i = batch
        pred = self.mod(x_i)
        loss = F.mse_loss(pred, y_i)
        self.log("train_loss", loss)
        
        if self.log_weight_stats:
            with torch.no_grad():
                weight_list=[(name, param) for name, param in self.mod.named_parameters() if name.split('.')[-1] == 'weight']
                for l in weight_list:
                    self.log(("train_mean"+l[0]), l[1].mean())
                    self.log(("train_std"+l[0]), l[1].std())        

        return(loss)
        
    def validation_step(self, batch, batch_idx):
        y_i, x_i = batch
        pred = self.mod(x_i)
        loss = F.mse_loss(pred, y_i)
        self.log('val_loss', loss)        
     
    def configure_optimizers(self, **kwargs):
        optimizer = torch.optim.Adam(self.parameters(), **kwargs)
        return optimizer