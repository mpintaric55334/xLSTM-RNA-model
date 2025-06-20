from model.dna_xlstm.xlstm.xlstm_lm_model import xLSTMLMModelConfig, xLSTMLMModel
import torch.nn as nn


class RIBOX(nn.Module):

    def __init__(self, cfg: xLSTMLMModelConfig):
        super().__init__()
        self.model = xLSTMLMModel(cfg)

    def forward(self, x):

        x = self.model(x)

        return x

    def headless_forward(self, x):

        x = self.model.headless_forward(x)

        return x
