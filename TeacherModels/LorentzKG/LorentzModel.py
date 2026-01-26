import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter

from manifolds.lorentz import Lorentz
import numpy as np


def othogonal_matrix(vv, I3, Iw):  # vv tensor of [#batch, dim-1, dim-1]
    bvv = torch.einsum('bwv, bwk -> bwvk', vv, vv)
    nbvv = torch.einsum('bwlv, bwvi -> bwli', vv.unsqueeze(-2), vv.unsqueeze(-1))
    qbvvt = (I3 - 2 * bvv / nbvv).permute([1, 0, 2, 3])
    for i in range(qbvvt.shape[0]):
        Iw = Iw @ qbvvt[i]
    return Iw  # [batch, dim-1, dim-1] othogonal matrix


class LorentzRotation(nn.Module):
    def __init__(self, manifold, num_emb, dim):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.num_emb = num_emb
        self.linear = nn.Embedding(num_emb, (dim - 1) * (
                    dim - 1))  # , max_norm=1.0 - 1e-4, norm_type=1, scale_grad_by_freq=True)  # this is the v-vector
        self.register_buffer('I3', torch.eye(self.dim - 1,).view(1, 1, self.dim - 1, self.dim - 1).repeat(
            [self.num_emb, self.dim - 1, 1, 1]))
        self.register_buffer('Iw', torch.eye(self.dim - 1,).view(1, self.dim - 1, self.dim - 1).repeat(
            [self.num_emb, 1, 1]))

    def forward(self, para):  # x, r, r_idx):
        # x [batch, n, dim]
        x = para[0]
        r_idx = para[1]
        x_0 = x.narrow(-1, 0, 1)  # [x_0] [batch, n, 1]
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)  # x_narrow = [x_1,...x_n] [batch, dim-1]
        ww = self.linear.weight
        ww = torch.nn.functional.gelu(ww)
        # do we need GELU here??
        ww = ww.view(-1, self.dim - 1, self.dim - 1)  # [num_rel, dim-1, dim-1]
        ww = othogonal_matrix(ww, self.I3, self.Iw)
        ww = ww[r_idx]  # batch dim-1 dim-1
        x_narrow = torch.einsum('bnd, bdc -> bnc', x_narrow, ww)
        xo = torch.cat([x_0, x_narrow], dim=-1)
        return (xo, r_idx)


class LorentzBoost2(nn.Module):
    def __init__(self, manifold, num_emb, dim):
        super().__init__()
        self.manifold = manifold
        self.num_emb = num_emb
        self.dim = dim
        self.linear = nn.Embedding(num_emb, dim - 1)
        self.clamp_max = 1e4
        self.clamp_min = -1e4

    def forward(self, para):  # x, r_idx):
        x = para[0]
        r_idx = para[1]
        r_o = self.linear(r_idx)
        r_o = torch.tanh(r_o)
        r_o = r_o / np.power(self.dim, 1)
        t = x.narrow(-1, 0, 1)  # first dim of lorentz vector (ct) [batch, len, 1]
        r = x.narrow(-1, 1, x.shape[-1] - 1)  # the remaining vectors of space portion in lorentz [batch, len, dim-1]
        zeta = 1 / (torch.sqrt(
            1 - torch.einsum('bld, bdi -> bli', r_o.unsqueeze(1), r_o.unsqueeze(-1))) + 1e-8)  # [batch, 1, 1]
        v2 = torch.einsum('bld, bdi -> bli', r_o.unsqueeze(1), r_o.unsqueeze(-1))  # [batch, 1, 1]
        r_o = r_o.unsqueeze(1)  # [batch, 1, dim-1]
        x_0 = zeta * t - zeta * torch.einsum('bld, bid -> bli', r, r_o)
        x_r = -1 * zeta * t * r_o + r + ((zeta - 1) / (v2 + 1e-9)) * torch.einsum('bldj, blj -> bld',
                                                                                torch.einsum('bld, blj -> bldj', r_o, r_o),
                                                                                r)
        xo = torch.cat([x_0, x_r], dim=-1)
        return (xo, r_idx)

class MarginLoss(nn.Module):
    def __init__(self, args):
        super(MarginLoss,self).__init__()
        self.args = args
        self.positive_weight = args.positive_weight
    
    def forward(self, intervals, targets=None):
        assert torch.all(targets[:, 0] == 1.0), \
            "CRRLoss Assumption Failed: Not all elements in the first column of targets are 1.0."
        
        # 验证2: 确保 targets 的其余列全部是 0.0
        assert torch.all(targets[:, 1:] == 0.0), \
            "CRRLoss Assumption Failed: Not all elements in the remaining columns of targets are 0.0."
        
        pos_scores = intervals[:, 0]  # shape: (batch_size,)
        neg_scores = intervals[:, 1:] # shape: (batch_size, num_neg)
        
        negative_score = F.logsigmoid(-neg_scores).mean(dim=-1)
        positive_score = F.logsigmoid(pos_scores)

        positive_sample_loss = -positive_score.mean()
        negative_sample_loss = -negative_score.mean()

        loss = self.positive_weight * positive_sample_loss + (1-self.positive_weight) * negative_sample_loss

        return loss


class CRRLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.t = args.CRR_t
        self.p = args.CRR_p

    def cliff_sigmod(self, x):
        exponent = (self.p - x) / self.t
        exponent = torch.clamp(exponent, min=-20, max=20)
        return 1 / (1 + torch.exp(exponent))

    def forward(self, intervals, targets=None): # targets 参数可以保留以兼容接口，但不再使用
        assert torch.all(targets[:, 0] == 1.0), \
            "CRRLoss Assumption Failed: Not all elements in the first column of targets are 1.0."
        
        assert torch.all(targets[:, 1:] == 0.0), \
            "CRRLoss Assumption Failed: Not all elements in the remaining columns of targets are 0.0."
        
        pos_scores = intervals[:, 0]  # shape: (batch_size,)
        neg_scores = intervals[:, 1:] # shape: (batch_size, num_neg)

        if pos_scores.numel() == 0:
            return torch.tensor(0.0, device=intervals.device)
            
        pos_scores = pos_scores.unsqueeze(1)
        diff = pos_scores - neg_scores
        sum_of_cliffs = torch.sum(self.cliff_sigmod(diff), axis=1)
        
        loss = (torch.log(sum_of_cliffs + 1.0)).mean()
        
        return loss

class BCELoss(nn.Module):
    def __init__(self, args):
        super(BCELoss, self).__init__()
        self.args = args
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.t = args.BCE_t
        if args.adv != 0.0:
            self.adv = args.adv
            self.adv_flag = True
        else:
            self.adv_flag = False            
    
    def calculate_weight(self, intervals):
        negative_score = intervals[:, 1:]
        negative_weights = F.softmax(negative_score * self.adv, dim=1).detach()
        positive_weights = torch.ones_like(intervals[:, 0:1]) 
        all_weights = torch.cat([positive_weights, negative_weights], dim=1)
        
        return all_weights
    
    def forward(self, intervals, targets):
        if self.adv_flag:
            weights = self.calculate_weight(intervals)
            intervals = intervals / self.t
            loss = F.binary_cross_entropy_with_logits(intervals, targets, weight=weights, reduction='mean')
        else:
            intervals = intervals / self.t
            loss = self.loss(intervals, targets)
        return loss


class HyperNet(nn.Module):
    def __init__(self, args, d, dims, max_norm, margin, neg_sample, npos, noise_reg):
        super(HyperNet, self).__init__()
        self.args = args
        self.manifold = Lorentz(max_norm=max_norm, k=args.Lorentz_k)  # , learnable=True)
        self.dim = dims
        self.noise_reg = noise_reg
        self.num_r_emb = len(d.relations)
        self.num_e_emb = len(d.entities)
        self.emb_entity_manifold = ManifoldParameter(self.manifold.random_normal((self.num_e_emb, dims),
                                                                                 std=1. / math.sqrt(dims)),
                                                     manifold=self.manifold, )
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(self.num_e_emb))
        self.bias_tail = torch.nn.Parameter(torch.zeros(self.num_e_emb))
        self.BCEloss = BCELoss(args)
        self.CRRloss = CRRLoss(args)
        self.Marginloss = MarginLoss(args)
        self.neg_sample = neg_sample
        self.npos = npos
        # below two can have different combinations to be "deep"

        self.head_linear = nn.Sequential(
            LorentzRotation(self.manifold, self.num_r_emb, dims ),
            LorentzBoost2(self.manifold, self.num_r_emb, dims ),
            # LorentzRotation(self.manifold, self.num_r_emb, dims ),
            # LorentzBoost2(self.manifold, self.num_r_emb, dims ),
            # ...
        )
        self.tail_linear = nn.Sequential(
            LorentzRotation(self.manifold, self.num_r_emb, dims),
            LorentzBoost2(self.manifold, self.num_r_emb, dims ),
        )

    def forward(self, u, r, v):
        if self.training:
            npos = v.shape[1]
            n1, p1 = None, None
            for i in range(npos):
                if len(u.shape) == 2:
                    u_idx = u[:, i]
                    t_idx = r[:, i]
                    v_idx = v[:, i, :]
                else:
                    u_idx = u[:, i, :]
                    t_idx = r[:, i]
                    v_idx = v[:, i]

                n_1 = self._forward(u_idx, t_idx, v_idx)
                if p1 is None:
                    p1 = n_1[:, 0:1]  # first record
                    n1 = n_1[:, 1:]
                else:
                    p1 = torch.cat([p1, n_1[:, 0:1]], dim=1)
                    n1 = torch.cat([n1, n_1[:, 1:]], dim=1)
                del n_1
            ndist = torch.cat([p1, n1], dim=1)  # [batch, npos + nneg*npos]
            del n1
            del p1
            return ndist
        else:
            return self._forward(u, r, v)


    def _forward(self, u_idx, r_idx, v_idx):
        h = self.emb_entity_manifold[u_idx]  # [batch,dim]
        t = self.emb_entity_manifold[v_idx]  # [batch,nneg+1,dim]
        if len(h.shape) == 2:
            h = h.unsqueeze(1)  # [batch, 1, dim]
            u_idx = u_idx.unsqueeze(1)
        elif len(t.shape) == 2:
            t = t.unsqueeze(1)
            v_idx = v_idx.unsqueeze(1)
        transformed_h, *_ = self.head_linear((h, r_idx))  # [batch, 1,  dim]
        transformed_t, *_ = self.tail_linear((t, r_idx))  # [batch, nneg+1, dim]
        mkv_interval = self.manifold.cinner2((transformed_t - transformed_h), (transformed_t - transformed_h)).squeeze()

        bias_terms = torch.tanh(self.bias_head[u_idx]) + torch.tanh(self.bias_tail[v_idx])
        if self.training:
            rnd_regular_head = self.noise_reg * torch.randn((mkv_interval.shape[0], 1), device=self.bias_head.get_device(), requires_grad=False)
            bias_terms += rnd_regular_head
        
        final_score = self.margin - mkv_interval + bias_terms
        
        return final_score
