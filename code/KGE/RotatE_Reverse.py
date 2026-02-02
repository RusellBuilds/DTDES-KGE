import logging
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class RotatE_Reverse(nn.Module):
    def __init__(self, args, model_type='Euc'):
        super(RotatE_Reverse, self).__init__()
        self.args = args
        self.model_type = model_type
        
        if model_type=='Euc':
            self.teacher_embedding_dim = self.args.feature_teacher_Euc_embedding_dim
            self.teacher_margin = self.args.feature_teacher_Euc_teacher_margin
            self.teacher_embedding_range = self.teacher_margin + 2.0
            pretrain_model = torch.load(os.path.join(self.args.feature_teacher_Euc_pretrain_path, 'checkpoint'),map_location='cpu')
            
        elif model_type == 'Hyper':
            self.teacher_embedding_dim = self.args.feature_teacher_Hyper_embedding_dim
            self.teacher_margin = args.feature_teacher_Hyper_teacher_margin
            self.teacher_embedding_range = self.teacher_margin + 2.0
            pretrain_model = torch.load(os.path.join(self.args.feature_teacher_Hyper_pretrain_path, 'checkpoint'),map_location='cpu')
            
        elif self.model_type == 'Stu':
            self.student_embedding_dim = self.args.student_rotate_embedding_dim
            self.student_margin = self.args.pos_gamma
            self.student_embedding_range = self.student_margin + 2.0
            
        if model_type=='Euc' or model_type == 'Hyper':
            if 'entity_embedding' in pretrain_model['model_state_dict']:
                self.entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu(), requires_grad=False)
                self.relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu(), requires_grad=False)
            elif 'EmbeddingManager.entity_embedding' in pretrain_model['model_state_dict']:
                self.entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu(), requires_grad=False)
                self.relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu(), requires_grad=False)
            else:
                raise ValueError(f"未知的entity_embedding以及relation_embedding位置: RotatE")
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(self.args.nentity, self.student_embedding_dim * 2))
            self.relation_embedding = nn.Parameter(torch.zeros(self.args.nrelation, self.student_embedding_dim))
            nn.init.uniform_(tensor=self.entity_embedding, a=-self.student_embedding_range/self.student_embedding_dim, b=self.student_embedding_range/self.student_embedding_dim)
            nn.init.uniform_(tensor=self.relation_embedding, a=-self.student_embedding_range/self.student_embedding_dim, b=self.student_embedding_range/self.student_embedding_dim)
        
        
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
        
        if self.model_type=='Euc' or self.model_type == 'Hyper':
            return self.tea_func(head, relation, tail)
        else:
            return self.stu_func(head, relation, tail)
    
    
    def get_embedding(self, sample):
        
        return self.sampling(sample) 
    
    
    def tea_func(self, head, relation, tail):
        
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/((self.teacher_embedding_range/self.teacher_embedding_dim)/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = score.sum(dim = 2)
        score = self.teacher_margin - score
        
        return score

    def stu_func(self, head, relation, tail):
        
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/((self.student_embedding_range/self.student_embedding_dim)/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = score.sum(dim = 2)
        score = self.student_margin - score
        
        return score

