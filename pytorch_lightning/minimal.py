import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
import torch.optim as optim

class LitModel():
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28*28, 1)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), 1)))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # forward mannully.
        loss = F.cross_entropy(y, y_hat)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters, lr=0.02)
        