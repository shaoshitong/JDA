import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function


def cross_entropy(pred, target):
    return -torch.mean(torch.sum(target * torch.log_softmax(pred, -1), -1))


def metric(l_s, l_t, label_s):
    l_s=F.normalize(l_s,dim=1,eps=1e-8)
    l_t=F.normalize(l_t,dim=1,eps=1e-8)
    y_sparse = torch.argmax(input=label_s, dim=1)
    equality_matrix = torch.eq(torch.reshape(y_sparse, [-1, 1]), y_sparse)
    equality_matrix = equality_matrix.long()
    p_target = (equality_matrix / torch.sum(equality_matrix, [1], keepdim=True))
    match_ab = torch.matmul(l_s, torch.transpose(l_t, 1, 0))
    # (S X T^T) X (T X S^T)
    p_ab = torch.softmax(match_ab, -1)
    p_ba = torch.softmax(torch.transpose(match_ab, 1, 0), -1)
    p_aba = torch.matmul(p_ab, p_ba)
    # walker_loss = F.cross_entropy(
    #     torch.log(1e-8 + p_aba), p_target)
    walker_loss = F.mse_loss(p_aba, p_target,reduction="mean",size_average=True)
    visit_probability = torch.mean(p_ab, [0], keepdim=True)
    _, t_nb = p_ab.shape
    # visit_loss = F.cross_entropy(torch.log(1e-8 + visit_probability), torch.ones([1, t_nb]).to(l_s.device) / t_nb)
    visit_loss = F.mse_loss(visit_probability, torch.ones([1, t_nb]).to(l_s.device) / t_nb,reduction="mean",size_average=False)
    per_row_accuracy = (1 - torch.sum(equality_matrix * p_aba, 1) ** 0.5).mean()
    estimate_error = 1. - per_row_accuracy
    return walker_loss, visit_loss, per_row_accuracy, estimate_error

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
    def forward(self,x):
        mean,std=x.mean([0],keepdim=True),x.std([0],keepdim=True)
        x=(x-mean)/(std+1e-8)
        return x

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()
        return output, None


class EmotionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionModel, self).__init__()
        self.num_classes = num_classes
        linear = nn.Linear(input_dim, 1024)
        nn.init.trunc_normal_(linear.weight.data, 0, 0.1)
        nn.init.constant_(linear.bias.data, 0.1)
        linear2 = nn.Linear(1024, num_classes)
        nn.init.trunc_normal_(linear2.weight.data, 0, 0.1)
        nn.init.constant_(linear2.weight.data, 0.1)
        linear3 = nn.Linear(1024, 2)
        nn.init.trunc_normal_(linear3.weight.data, 0, 0.1)
        nn.init.constant_(linear3.weight.data, 0.1)
        self.blocks = nn.Sequential()
        self.blocks.add_module("linear", linear)
        self.blocks.add_module("relu", nn.ReLU(inplace=True))
        self.blocks.add_module("dropout", nn.Dropout(p=0.1))
        # self.blocks.add_module("bn",Norm())
        self.linear1 = linear
        self.linear2 = linear2
        self.linear3 = linear3
        self.decay = 0.99
        self.per_row_accuracy = 0.
        self.estimate_error = 0.
        self.alpha = 1.0
        self.cross_entropy = nn.CrossEntropyLoss()
        self.pred_loss = 0.
        self.domain_loss = 0.
        self.walker_loss = 0.
        self.visit_loss = 0.

    def forward(self, x):
        features = self.blocks(x)
        if self.training:
            logits = self.linear2(features[:features.shape[0] // 2])
            with torch.no_grad():
                target_logits = self.linear2(features[features.shape[0] // 2:])

        else:
            logits = self.linear2(features[features.shape[0] // 2:])
            target_logits = logits
        feat = ReverseLayerF.apply(features, self.alpha)
        domain_pred = self.linear3(feat)
        return features, logits, domain_pred, target_logits

    def fit_one_step(self, X, y, training=True):
        X = X.view(X.shape[0], -1)
        y = y.view(y.shape[0], -1)
        if y.ndim == 1:
            y = torch.zeros(y.shape[0], self.num_classes).scatter_(1, y.unsqueeze(-1), 1)
        train_b = X.shape[0] // 2
        if training == True:
            self.train()
        else:
            self.eval()
        features, logits, domain_pred, target_logits = self(X)
        self.pred_loss = self.cross_entropy(logits[:train_b], y[:train_b])
        self.acc_nums=(torch.argmax(logits[:train_b],1)==torch.argmax(y[:train_b],1)).sum()
        self.target_logits = target_logits
        self.target_y = y[train_b:]
        self.walker_loss, self.visit_loss, per_row_accuracy, estimate_error = metric(features[:train_b],
                                                                                     features[train_b:], y[:train_b])
        self.per_row_accuracy = self.decay * self.per_row_accuracy + (1 - self.decay) * per_row_accuracy.item()
        self.estimate_error = self.decay * self.estimate_error + (1 - self.decay) * estimate_error.item()
        source_domain = torch.tensor([[1., 0.]], requires_grad=True).to(features.device).repeat([train_b, 1])
        target_domain = torch.tensor([[0., 1.]], requires_grad=True).to(features.device).repeat([train_b, 1])
        domain_target = torch.cat([source_domain, target_domain], dim=0)
        self.domain_loss = F.cross_entropy(domain_pred, domain_target)
        self.l2_loss = torch.sqrt(torch.sum(self.linear1.weight ** 2)) / 2 + torch.sqrt(
            torch.sum(self.linear2.weight ** 2)) / 2 + torch.sqrt(torch.sum(self.linear3.weight ** 2)) / 2
        self.l2_loss = 0.001 * self.l2_loss
