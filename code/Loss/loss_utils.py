import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DistributionSelector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_dim = 6
        self.output_dim = 2
        self.hidden_dim1 = args.DistributionSelector_hidden_dim1
        self.hidden_dim2 = args.DistributionSelector_hidden_dim2
        self.eps = 1e-6

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.GELU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.GELU(),
            nn.Linear(self.hidden_dim2, self.output_dim)
        )

    def forward(self, PT1_score, PT2_score):
        mean1 = torch.mean(PT1_score, dim=1, keepdim=True).detach()
        std1 = torch.std(PT1_score, dim=1, keepdim=True).detach()

        mean2 = torch.mean(PT2_score, dim=1, keepdim=True).detach()
        std2 = torch.std(PT2_score, dim=1, keepdim=True).detach()

        mean_diff = mean1 - mean2
        std_ratio = std1 / (std2 + self.eps)
        inputs = torch.cat([mean1, mean2, std1, std2, mean_diff, std_ratio], dim=1)
        raw_output = self.MLP(inputs)
        # 原代码: mean_star = raw_output[:, 0:1]
        # 因为后面算KL散度需要过softmax，所以这里可以把均值都设为0，因为softmax对加法不敏感，只需要std起作用
        # 但是如果想尝试用Huber Loss蒸馏的话，这里可以使用学出来的mean_star
        mean_star = torch.zeros_like(raw_output[:, 0:1]) 
        std_star_raw = raw_output[:, 1:2]
        std_star = F.softplus(std_star_raw) + self.eps

        return mean_star, std_star




class weight_learner(nn.Module):
    def __init__(self, args):
        super(weight_learner, self).__init__()
        self.args = args
        self.entity_dim = args.target_entity_dim
        self.relation_dim = args.target_relation_dim
        self.layer_mul = args.weightlearner_layer_mul
        self.hidden_dim = args.weightlearner_hidden_dim
        
        self.entity_transfer = nn.Sequential(
            nn.Linear(self.entity_dim, self.entity_dim * self.layer_mul),
            nn.GELU(),
            nn.Linear(self.entity_dim * self.layer_mul, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        self.relation_transfer = nn.Sequential(
            nn.Linear(self.relation_dim, self.relation_dim * self.layer_mul),
            nn.GELU(),
            nn.Linear(self.relation_dim * self.layer_mul, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.teacher_message_transfer = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, self.hidden_dim),
        )
        self.student_message_transfer = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, self.hidden_dim),
        )
        

        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4 * self.layer_mul),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4 * self.layer_mul, 2)
        )


    def _cal_tea_message(self, PT1_score, PT2_score):
        PT1_score_d = PT1_score.detach()
        PT2_score_d = PT2_score.detach()
        
        log_PT1_prob = F.log_softmax(PT1_score_d, dim=-1)
        log_PT2_prob = F.log_softmax(PT2_score_d, dim=-1)

        PT1_prob = torch.exp(log_PT1_prob)
        PT2_prob = torch.exp(log_PT2_prob)

        kl_div_p2_p1 = F.kl_div(log_PT1_prob, PT2_prob, reduction='none').sum(dim=-1, keepdim=True)
        kl_div_p1_p2 = F.kl_div(log_PT2_prob, PT1_prob, reduction='none').sum(dim=-1, keepdim=True)
        
        PT1_pos = PT1_prob[:, 0:1]
        PT2_pos = PT2_prob[:, 0:1]

        teacher_message = torch.cat([PT1_pos, PT2_pos, kl_div_p1_p2, kl_div_p2_p1], dim=-1)
        return teacher_message
    
    def _cal_stu_message(self, stu_score, PT1_score, PT2_score):
        stu_score_d = stu_score.detach()
        PT1_score_d = PT1_score.detach()
        PT2_score_d = PT2_score.detach()

        log_stu_prob = F.log_softmax(stu_score_d, dim=-1)
        stu_prob = torch.exp(log_stu_prob)
        
        PT1_prob = F.softmax(PT1_score_d, dim=-1)
        PT2_prob = F.softmax(PT2_score_d, dim=-1)
        
        kl_div_t1_stu = F.kl_div(log_stu_prob, PT1_prob, reduction='none').sum(dim=-1, keepdim=True)
        kl_div_t2_stu = F.kl_div(log_stu_prob, PT2_prob, reduction='none').sum(dim=-1, keepdim=True)

        stu_pos = stu_prob[:, 0:1]

        stu_message = torch.cat([stu_pos, kl_div_t1_stu, kl_div_t2_stu], dim=-1)
        
        return stu_message
    
    def forward(self, head, relation, tail=None, stu_score=None, PT1_score=None, PT2_score=None, data=None):
        head_transfer = self.entity_transfer(head)
        relation_transfer = self.relation_transfer(relation)
        
        teacher_message = self._cal_tea_message(PT1_score.detach(), PT2_score.detach())
        teacher_message_transfer = self.teacher_message_transfer(teacher_message)
        
        stu_message = self._cal_stu_message(stu_score.detach(), PT1_score.detach(), PT2_score.detach())
        stu_message_transfer = self.student_message_transfer(stu_message)
        
        teacher_message_transfer = teacher_message_transfer.unsqueeze(1)
        stu_message_transfer = stu_message_transfer.unsqueeze(1)

        combined = torch.cat([head_transfer, relation_transfer, teacher_message_transfer, stu_message_transfer], dim=2)
        x = self.MLP(combined)

        return x


class Prior_Weight(nn.Module):
    
    def __init__(self, args):
        super(Prior_Weight, self).__init__()
        self.args = args

        if 'FB15k-237' == self.args.dataset:
            cur_path = self.args.data_path + '/FB15k-237_relation_cur.txt'
            kra_path = self.args.data_path + '/FB15k-237_relation_kra.txt'
        elif 'WN18RR' == self.args.dataset:
            cur_path = self.args.data_path + '/wn18rr_relation_cur.txt'
            kra_path = self.args.data_path + '/wn18rr_relation_kra.txt'
        
        self.cur_dict = self.read_file_to_dict(cur_path)
        self.kra_dict = self.read_file_to_dict(kra_path)
        
        num_rel = self.args.nrelation       
        self.device = torch.device('cuda', int(self.args.gpu_id))     

        self.cur = torch.zeros(num_rel, device=self.device)
        self.kra = torch.zeros(num_rel, device=self.device)
        for k, v in self.cur_dict.items():
            self.cur[k] = v
            self.cur[k + num_rel//2] = v
        for k, v in self.kra_dict.items():
            self.kra[k] = v
            self.kra[k + num_rel//2] = v
        self.cur.requires_grad_(False)
        self.kra.requires_grad_(False)


    def read_file_to_dict(self, path: str):
        data = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue   
                first, second, *_ = line.split() 
                data[int(first)] = round(float(second), 1)
        return data


    def cal_prior(self, data):
        cur_thres = self.args.cur_thres
        cur_opt   = self.args.cur_opt
        kra_thres = self.args.kra_thres
        kra_opt   = self.args.kra_opt

        positive_sample, negative_sample = data
        r_idx = positive_sample[:, 1]
        
        cur = self.cur[r_idx]     # [batch]
        kra = self.kra[r_idx]     # [batch]
        batch = cur.size(0)

        prior_weight = torch.empty(
        batch, 2,
        dtype=cur.dtype,
        device=self.device,
        requires_grad=False
        )

        mask_cur_low   = cur < cur_thres       
        mask_cur_high = ~mask_cur_low          

        prior_weight[mask_cur_low, 1] = cur_opt    
        prior_weight[mask_cur_low, 0] = 1 - cur_opt   

        prior_weight[mask_cur_high, 0] = cur_opt
        prior_weight[mask_cur_high, 1] = 1 - cur_opt

        mask_kra_high = kra > kra_thres   
        mask_kra_low = ~mask_kra_high
        prior_weight[mask_kra_high, 1] += kra_opt
        prior_weight[mask_kra_high, 0] += 1 - kra_opt
        
        prior_weight[mask_kra_low, 0] += kra_opt
        prior_weight[mask_kra_low, 1] += 1 - kra_opt

        prior_weight = prior_weight/2
        
        return prior_weight.unsqueeze(1)
  
    def forward(self,data):
        weights = self.cal_prior(data)
        return weights


