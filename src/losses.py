import torch.nn as nn

class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, reweight=True, alpha=None, num_classes=2):
        super(nn.CrossEntropyLoss, self).__init__()
        self.reweight = reweight
        self.alpha = alpha
        self.num_classes = num_classes

    def softmax(self, outputs):
        return (torch.exp(outputs.t()) / torch.sum(torch.exp(outputs), dim=1)).t()

    def encode(self, targets):
        encoded_targets = torch.zeros(targets.size(0), self.num_classes)
        encoded_targets.scatter_(1, targets.view(-1, 1).long(), 1).float()
        return encoded_targets

    def weights(self, outputs, targets):
        softmax_outputs = self.softmax(outputs)
        #if softmax_outputs.shape != targets.shape:
        encoded_targets = self.encode(targets)
        weights = softmax_outputs.gather(1, torch.argmax(encoded_targets, dim=1).unsqueeze(1)).squeeze(1) - softmax_outputs.gather(1, torch.argmax(softmax_outputs, dim=1).unsqueeze(1)).squeeze(1)/np.exp(self.alpha)
        return weights

    def forward(self, outputs, targets):
        softmax_outputs = self.softmax(outputs)
        loss = torch.sum(torch.log(softmax_outputs) * (targets), dim=1)
        if self.reweight:
            weighted_loss = - self.weights(outputs, targets) * loss
            return weighted_loss.mean()
        else:
            return loss.mean()

class WeightedFocalLoss(nn.CrossEntropyLoss):
    def __init__(self, reweight=True, alpha=None, gamma=None, num_classes=2):
        super(nn.CrossEntropyLoss, self).__init__()
        self.reweight = reweight
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def weights(self, outputs, targets):
        softmax_outputs = self.softmax(outputs)
        #if softmax_outputs.shape != targets.shape:
        encoded_targets = self.encode(targets)
        weights = softmax_outputs.gather(1, torch.argmax(encoded_targets, dim=1).unsqueeze(1)).squeeze(1) - softmax_outputs.gather(1, torch.argmax(softmax_outputs, dim=1).unsqueeze(1)).squeeze(1)/np.exp(self.alpha)
        return weights

    def forward(self, outputs, targets):
        criterion = WeightedCrossEntropyLoss(reweight=reweight, alpha=None, num_classes=2)
        cross_entropy_loss = criterion(outputs, targets)
        focal_loss = (1 - torch.exp(- cross_entropy_loss)) ** self.gamma * cross_entropy_loss
        if self.reweight:
            weighted_focal_loss = - self.weights(outputs, targets) * focal_loss
            return weighted_focal_loss.mean()
        else:
            return focal_loss.mean()
