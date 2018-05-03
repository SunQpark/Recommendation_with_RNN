import torch.nn.functional as F


def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)


# todo: sampling of positive, negative samples
def bayesian_personalized_ranking(y_input, y_target):
    pass
    
def top1_loss(y_input, y_target):
    pass