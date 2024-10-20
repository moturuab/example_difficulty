import torch.nn as nn
import torch
import numpy as np

class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, reweight=True, alpha=None, beta=None, delta=None, num_classes=2, warmup=0, device=None):
        super(nn.CrossEntropyLoss, self).__init__()
        self.reweight = reweight
        self.alpha = alpha
        self.beta=beta
        self.delta = delta
        self.warmup = warmup
        self.num_classes = num_classes
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def softmax(self, outputs):
        return (torch.exp(outputs.t()) / torch.sum(torch.exp(outputs), dim=1)).t()

    def encode(self, targets):
        encoded_targets = torch.zeros(targets.size(0), self.num_classes).to(self.device)
        encoded_targets.scatter_(1, targets.view(-1, 1).long(), 1).float()
        return encoded_targets

    def weights(self, correct_outputs, max_outputs):
        weights = (self.sigmoid(self.alpha*correct_outputs - max_outputs) + 
            self.sigmoid(-(self.beta*correct_outputs - max_outputs)) + 
            torch.exp(-(-(self.delta*correct_outputs - max_outputs))**2/2))
        return weights

    def forward(self, outputs, targets, epoch=-1):
        softmax_outputs = self.softmax(outputs)
        encoded_targets = self.encode(targets)
        loss = - torch.sum(torch.log(softmax_outputs) * (encoded_targets), dim=1)

        correct_outputs = softmax_outputs.gather(1, torch.argmax(encoded_targets, dim=1).unsqueeze(1)).squeeze(1)
        max_outputs = softmax_outputs.gather(1, torch.argmax(softmax_outputs, dim=1).unsqueeze(1)).squeeze(1)

        if self.reweight and epoch > self.warmup:
            weights = self.weights(correct_outputs, max_outputs)
            weighted_loss = weights * loss
            return correct_outputs, max_outputs, weights, weighted_loss.mean()
        else:
            return correct_outputs, max_outputs, None, loss.mean()

class WeightedFocalLoss(nn.CrossEntropyLoss):
    def __init__(self, reweight=True, alpha=None, beta=None, delta=None, gamma=None, num_classes=2, warmup=0, device=None):
        super(nn.CrossEntropyLoss, self).__init__()
        self.reweight = reweight
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.warmup = warmup
        self.num_classes = num_classes
        self.device = device

    def forward(self, outputs, targets, epoch=-1):
        criterion = WeightedCrossEntropyLoss(reweight=False, alpha=self.alpha, beta=self.beta, delta=self.delta, num_classes=self.num_classes, device=self.device)
        correct_outputs, max_outputs, weights, cross_entropy_loss = criterion(outputs, targets, epoch=epoch)
        focal_loss = (1 - torch.exp(- cross_entropy_loss)) ** self.gamma * cross_entropy_loss
        encoded_targets = criterion.encode(targets)
        if self.reweight and epoch > self.warmup:
            weights = criterion.weights(outputs, encoded_targets)
            weighted_focal_loss = weights * focal_loss
            return correct_outputs, max_outputs, weights, weighted_focal_loss.mean()
        else:
            return correct_outputs, max_outputs, None, focal_loss.mean()
