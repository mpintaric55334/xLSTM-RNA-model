import torch.nn as nn


class Loss:

    def __init__(self):

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, predictions, true_sequence, masks):

        loss = self.criterion(predictions, true_sequence)
        loss = loss * masks
        return loss.sum() / (masks.sum() + 1e-8)
