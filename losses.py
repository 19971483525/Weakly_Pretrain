"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class WeaklyConLoss(nn.Module):
    """Weaklyervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(WeaklyConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        # mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=device), mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(DenseContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        features = features.view(features.shape[0], features.shape[1], features.shape[2], -1)  # [32, 2, 128, 49]

        batch_size = features.shape[0]  # 32
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # [32, 32]
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)  # [32, 1]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # [32, 32]
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [64, 128, 49]
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # [32, 128, 49]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [64, 128, 49]
            anchor_count = contrast_count  # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits using Frobenius norm
        def frobenius_norm(tensor1, tensor2):
            return torch.sum((tensor1 - tensor2) ** 2, dim=[2, 3])
        
        # 计算 anchor_dot_contrast
        anchor_dot_contrast = -frobenius_norm(
            anchor_feature.unsqueeze(1),  # [64, 1, 128, 49]
            contrast_feature.unsqueeze(0)  # [1, 64, 128, 49]
        ) / self.temperature  # 结果形状为 [64, 64]

        # 数值稳定性处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 处理掩码和计算 log_prob
        mask = mask.repeat(anchor_count, contrast_count)  # [64, 64]
        logits_mask = torch.ones_like(logits)  # [64, 64]
        logits_mask = torch.scatter(
            logits_mask,
            1,
            torch.arange(logits.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # [64, 64]

        exp_logits = torch.exp(logits) * logits_mask  # [64, 64]
        # 添加一个小常数以避免数值问题
        safe_exp_logits_sum = exp_logits.sum(1, keepdim=True) + 1e-10
        log_prob = logits - torch.log(safe_exp_logits_sum)  # [64, 64]

        # 计算平均对数似然损失
        mask_pos_pairs = mask.sum(1)  # [64]
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype, device=device), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs  # [64]

        # 计算最终损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CombinedContrastiveLoss(nn.Module):
    def __init__(self, global_temperature=0.1, dense_temperature=0.1, lambda_weight=0.5):
        super(CombinedContrastiveLoss, self).__init__()
        self.global_temperature = global_temperature
        self.dense_temperature = dense_temperature
        self.lambda_weight = lambda_weight
        self.global_loss_fn = WeaklyConLoss(temperature=global_temperature)
        self.dense_loss_fn = DenseContrastiveLoss(temperature=dense_temperature)

    def forward(self, global_features, dense_features, labels=None, mask=None, positive_pairs=None, negative_pairs=None):
        global_loss = self.global_loss_fn(global_features, labels, mask)
        dense_loss = self.dense_loss_fn(dense_features)

        combined_loss = (1 - self.lambda_weight) * global_loss + self.lambda_weight * dense_loss

        return combined_loss, global_loss, dense_loss