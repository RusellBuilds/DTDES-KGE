import logging
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Combine_hr(nn.Module):
    def __init__(self, args):
        super(Combine_hr, self).__init__()
        self.args = args
        
        self.entity_dim = args.target_entity_dim
        self.relation_dim = args.target_relation_dim
        self.hidden_dim = args.combine_hr_hidden_dim
        self.layer_mul = args.combine_hr_layer_mul
        
        input_dim = self.entity_dim + self.relation_dim
        
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, input_dim * self.layer_mul),
            nn.GELU(),
            nn.Linear(input_dim * self.layer_mul, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim)  
        )
    
    def forward(self, eh, er):
        if eh.shape[1] != er.shape[1]:
            er = er.expand(-1, eh.shape[1], -1)
        
        combined = torch.cat((eh, er), dim=2)  # Shape: [batch, seq_len, entity_dim + relation_dim]
        

        batch_size, seq_len, feature_dim = combined.size()
        combined_reshaped = combined.view(batch_size * seq_len, -1)
        
        output_reshaped = self.MLP(combined_reshaped)
        
        output = output_reshaped.view(batch_size, seq_len, self.hidden_dim)
        
        return output


class BN(nn.Module):

    def __init__(self, num_features):
        super(BN, self).__init__()
        self.BatchNorm = nn.BatchNorm1d(num_features)
    
    def forward(self, t):
        batch_size, seq_len, num_features = t.size()

        t_reshaped = t.view(batch_size * seq_len, num_features)
        
        bn_output = self.BatchNorm(t_reshaped)
        
        output = bn_output.view(batch_size, seq_len, num_features)
        
        return output


class SCCF_Reverse(nn.Module):
    def __init__(self, args):
        super(SCCF_Reverse, self).__init__()
        self.args = args
        self.entity_embedding_dim = self.args.target_entity_dim
        self.relation_embedding_dim = self.args.target_relation_dim
        self.tau = self.args.SCCF_tau
        self.SCCF_mode = self.args.SCCF_mode
        
        self.combine_hr = Combine_hr(args)
        self.BN = BN(self.args.target_entity_dim)
        
        if self.args.init_checkpoint_path != '':
            pretrain_model = torch.load(os.path.join(self.args.init_checkpoint_path, 'checkpoint'))
            self.entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu(), requires_grad=False)
            self.relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu(), requires_grad=False)
        else:
            self.entity_embedding = nn.Parameter(torch.empty(self.args.nentity, self.entity_embedding_dim), requires_grad=True)
            self.relation_embedding = nn.Parameter(torch.empty(self.args.nrelation, self.relation_embedding_dim), requires_grad=True)
            nn.init.xavier_uniform_(self.entity_embedding)
            nn.init.xavier_uniform_(self.relation_embedding)
    
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
    
    
    def cosine_similarity(self, ehr, et):
        if ehr.shape[1] < et.shape[1]: 
            ehr = ehr.expand(-1, et.shape[1], -1)
        else:
            et = et.expand(-1, ehr.shape[1], -1)
        sim = F.cosine_similarity(ehr, et, dim=-1)
        return sim
    
    def SCCF_similarity1(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)
        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 3 / self.temperature) - torch.exp((dot_product / norm_product) ** 3 / self.temperature)
        return sim
    
    def SCCF_similarity3(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)
        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 3 / self.temperature)
        return sim
    
    def similarity1(self, ehr, et):
        return self.cosine_similarity(ehr, et)
    
    def similarity3(self, ehr, et):
        return self.cosine_similarity(ehr, et) + (self.cosine_similarity(ehr, et))**3
    
    
    def func(self, head, relation, tail):
        
        ehr = self.combine_hr(head, relation)
        et = self.BN(tail)
        
        if self.SCCF_mode == 'SCCF_similarity1':
            return self.SCCF_similarity1(ehr, et)
        elif self.SCCF_mode == 'SCCF_similarity3':
            return self.SCCF_similarity3(ehr, et)
        elif self.SCCF_mode == 'similarity1':
            return self.similarity1(ehr, et)
        elif self.SCCF_mode == 'similarity3':
            return self.similarity3(ehr, et)
        else:
            raise ValueError(f"未知的 SCCF_mode: {self.SCCF_mode}")


