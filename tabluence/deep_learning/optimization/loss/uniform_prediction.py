import torch
import torch.nn


class NonUniformPenalty(torch.nn.Module):
    def __init__(self):
        super(NonUniformPenalty, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, logits, targets):
        number_of_classes = logits.shape[1]
        device = logits.device
        prior = torch.ones(number_of_classes, device=device) / number_of_classes
        pred_mean = self.softmax(logits).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        return penalty
