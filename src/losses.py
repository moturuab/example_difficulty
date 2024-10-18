import torch.nn as nn
import torch
import numpy as np

class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, reweight=True, alpha=None, beta=None, num_classes=2, device=None):
        super(nn.CrossEntropyLoss, self).__init__()
        self.reweight = reweight
        self.alpha = alpha
        self.beta=beta
        self.num_classes = num_classes
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def softmax(self, outputs):
        return (torch.exp(outputs.t()) / torch.sum(torch.exp(outputs), dim=1)).t()

    def encode(self, targets):
        encoded_targets = torch.zeros(targets.size(0), self.num_classes).to(self.device)
        encoded_targets.scatter_(1, targets.view(-1, 1).long(), 1).float()
        return encoded_targets

    def weights(self, outputs, encoded_targets, m=0):
        softmax_outputs = self.softmax(outputs)
        correct_outputs = softmax_outputs.gather(1, torch.argmax(encoded_targets, dim=1).unsqueeze(1)).squeeze(1)
        max_outputs = softmax_outputs.gather(1, torch.argmax(softmax_outputs, dim=1).unsqueeze(1)).squeeze(1)
        if not m:
            print('ALPHA')
            print(torch.min(self.alpha*correct_outputs - max_outputs))
            print(torch.max(self.alpha*correct_outputs - max_outputs))
            #weights = self.sigmoid(self.alpha*correct_outputs - max_outputs + self.alpha)**self.alpha
            weights = self.sigmoid(self.alpha*correct_outputs - max_outputs)**self.alpha
            print(torch.min(weights))
            print(torch.max(weights))
        else:
            print('BETA')
            print(torch.min(-(self.beta*correct_outputs - max_outputs)))
            print(torch.max(-(self.beta*correct_outputs - max_outputs)))
            #weights = self.sigmoid(-(self.beta*correct_outputs - max_outputs) + self.beta)**self.beta
            weights = self.sigmoid(-(self.beta*correct_outputs - max_outputs))**self.beta
            print(torch.min(weights))
            print(torch.max(weights))
            #weights = torch.where(weights > 0.5, 0, 1-weights)
        return weights

    def forward(self, outputs, targets, m=0):
        softmax_outputs = self.softmax(outputs)
        encoded_targets = self.encode(targets)
        loss = - torch.sum(torch.log(softmax_outputs) * (encoded_targets), dim=1)
        if self.reweight:
            weights = self.weights(outputs, encoded_targets, m=m)
            weighted_loss = weights * loss
            return weighted_loss.mean()
        else:
            return loss.mean()

class WeightedFocalLoss(nn.CrossEntropyLoss):
    def __init__(self, reweight=True, alpha=None, beta=None, gamma=None, num_classes=2, device=None):
        super(nn.CrossEntropyLoss, self).__init__()
        self.reweight = reweight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.device = device

    def weights(self, outputs, encoded_targets, m=0):
        softmax_outputs = self.softmax(outputs)
        correct_outputs = softmax_outputs.gather(1, torch.argmax(encoded_targets, dim=1).unsqueeze(1)).squeeze(1)
        max_outputs = softmax_outputs.gather(1, torch.argmax(softmax_outputs, dim=1).unsqueeze(1)).squeeze(1)
        if not m:
            weights = self.sigmoid(self.alpha*correct_outputs - max_outputs)
        else:
            weights = self.sigmoid(-(self.beta*correct_outputs - max_outputs))
        return weights

    def forward(self, outputs, targets, m=0):
        criterion = WeightedCrossEntropyLoss(reweight=self.reweight, alpha=self.alpha, beta=self.beta, num_classes=self.num_classes, device=self.device)
        cross_entropy_loss = criterion(outputs, targets, m=m)
        focal_loss = (1 - torch.exp(- cross_entropy_loss)) ** self.gamma * cross_entropy_loss
        encoded_targets = criterion.encode(targets)
        if self.reweight:
            weighted_focal_loss = self.weights(outputs, encoded_targets, m=m) * focal_loss
            return weighted_focal_loss.mean()
        else:
            return focal_loss.mean()
