import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os


class HAKE_Reverse(nn.Module):
    def __init__(self, args, model_type='Euc'):
        super(HAKE_Reverse, self).__init__()
        self.args = args
        
        if model_type=='Euc':
            self.hidden_dim = args.logits_teacher_Euc_embedding_dim
            pretrain_model = torch.load(os.path.join(self.args.logits_teacher_Euc_pretrain_path, 'checkpoint'), map_location='cpu')
        
        self.gamma  = nn.Parameter(pretrain_model['model_state_dict']['gamma'].cpu(), requires_grad=False)
        self.embedding_range = nn.Parameter(pretrain_model['model_state_dict']['embedding_range'].cpu(), requires_grad=False)
        self.entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu(), requires_grad=False)
        self.relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu(), requires_grad=False)
        self.phase_weight = nn.Parameter(pretrain_model['model_state_dict']['phase_weight'].cpu(), requires_grad=False)
        self.modulus_weight = nn.Parameter(pretrain_model['model_state_dict']['modulus_weight'].cpu(), requires_grad=False)

        self.pi = 3.14159262358979323846
        self.epsilon = 2.0


    def sampling(self, sample):
        head_part, tail_part = sample
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

        head = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=head_part[:, 0]
        ).unsqueeze(1)

        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=head_part[:, 1]
        ).unsqueeze(1)

        pos_tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=head_part[:, 2]
        ).unsqueeze(1)
        
        neg_tail = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=tail_part.view(-1)
        ).view(batch_size, negative_sample_size, -1)
        
        tail = torch.cat((pos_tail, neg_tail), dim=1)
        
        return head, relation, tail
    
    def forward(self, sample):
        head, relation, tail = self.sampling(sample)
        
        return self.func(head, relation, tail)
    
    
    def get_embedding(self, sample):
        
        return self.sampling(sample)     
    
    
    def func(self, head, rel, tail):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        score = self.gamma.item() - (phase_score + r_score)
        
        return score