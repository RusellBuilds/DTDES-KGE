import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

original_directory = os.getcwd()
new_directory = original_directory + '/code/Loss/'
sys.path.append(new_directory)
from loss_utils import DistributionSelector, weight_learner, Prior_Weight


class SimpleSigmoidLoss(nn.Module):
    def __init__(self, args):
        super(SimpleSigmoidLoss, self).__init__()
        self.args = args
        if self.args.pos_neg_gamma_equal:
            self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
            self.neg_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
        else:
            self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
            self.neg_margin = nn.Parameter(torch.Tensor([self.args.neg_gamma]))
        
        self.pos_margin.requires_grad = False
        self.neg_margin.requires_grad = False
    
    def forward(self, stu_score, subsampling_weight=None):
        student_p_score, student_n_score = stu_score[:, 0], stu_score[:, 1:]
        
        negative_score = F.logsigmoid(-student_n_score).mean(dim = 1)
        positive_score = F.logsigmoid(student_p_score)
        
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        
        loss = (positive_sample_loss + negative_sample_loss)/2
        
        loss_record = {
            'Sigmoid_hard_positive_sample_loss': positive_sample_loss.item(),
            'Sigmoid_hard_negative_sample_loss': negative_sample_loss.item(),
            'Sigmoid_hard_loss': loss.item(),
        }
            
        return loss, loss_record


class SigmoidLoss(nn.Module):
    def __init__(self, args, who_use='stu'):
        super(SigmoidLoss, self).__init__()
        self.args = args
        
        # Margin的取值
        if who_use == 'stu': 
            if self.args.pos_neg_gamma_equal: 
                self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
                self.neg_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
            else: 
                self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
                self.neg_margin = nn.Parameter(torch.Tensor([self.args.neg_gamma]))
            self.pos_margin.requires_grad = False
            self.neg_margin.requires_grad = False
        else: 
            if self.args.tea_stu_gamma_equal: 
                self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
                self.neg_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
            elif self.args.tea_pos_neg_gamma_equal: 
                self.pos_margin = nn.Parameter(torch.Tensor([self.args.tea_pos_gamma]))
                self.neg_margin = nn.Parameter(torch.Tensor([self.args.tea_pos_gamma]))
            else:
                self.pos_margin = nn.Parameter(torch.Tensor([self.args.tea_pos_gamma]))
                self.neg_margin = nn.Parameter(torch.Tensor([self.args.tea_neg_gamma]))
            self.pos_margin.requires_grad = False
            self.neg_margin.requires_grad = False
        
        if who_use == 'stu': 
            if self.args.learn_hard_positive_weight: 
                self.hard_positive_weight = nn.Parameter(torch.Tensor([0]))
                self.hard_positive_weight.requires_grad = True
            else: 
                self.hard_positive_weight = nn.Parameter(torch.Tensor([self.args.hard_positive_weight]))
                self.hard_positive_weight.requires_grad = False
        else: 
            if self.args.tea_learn_hard_positive_weight: 
                self.hard_positive_weight = nn.Parameter(torch.Tensor([0]))
                self.hard_positive_weight.requires_grad = True
            else: 
                if self.args.tea_stu_positive_weight_equal: 
                    self.hard_positive_weight = nn.Parameter(torch.Tensor([self.args.hard_positive_weight]))
                    self.hard_positive_weight.requires_grad = False
                else: 
                    self.hard_positive_weight = nn.Parameter(torch.Tensor([self.args.tea_hard_positive_weight]))
                    self.hard_positive_weight.requires_grad = False                    

        if self.args.negative_adversarial_sampling:
            self.adv_temperature = nn.Parameter(torch.Tensor([self.args.adversarial_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def forward(self, similarity, teacher_score=None, subsampling_weight=None):
        student_p_score, student_n_score = similarity[:, 0], similarity[:, 1:]
        mask = torch.ones_like(student_n_score, dtype=torch.bool, device=student_n_score.device)

        if not self.args.execute_ditill_teacher_to_rotate:
            p_score_margin = student_p_score - self.pos_margin
            n_score_margin = student_n_score - self.neg_margin
        else:
            p_score_margin = student_p_score
            n_score_margin = student_n_score
            
        
        if self.adv_flag:
            softmax_weights = F.softmax(student_n_score * self.adv_temperature, dim=1).detach()
            logsigmoid_values = F.logsigmoid(-n_score_margin)
            
            weighted_losses = softmax_weights * logsigmoid_values
            masked_weighted_losses = weighted_losses * mask.float()
            negative_score = masked_weighted_losses.sum(dim=1)
        else:
            logsigmoid_values = F.logsigmoid(-n_score_margin)
            masked_logsigmoid_values = logsigmoid_values * mask.float()

            num_unfiltered_samples = mask.sum(dim=1).float()
            num_unfiltered_samples[num_unfiltered_samples == 0] = 1.0 
            
            negative_score = masked_logsigmoid_values.sum(dim=1) / num_unfiltered_samples

        positive_score = F.logsigmoid(p_score_margin)

        if not self.args.subsampling:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        

        if self.args.learn_hard_positive_weight:
            positive_weight = torch.sigmoid(self.hard_positive_weight)
        else:
            positive_weight = self.hard_positive_weight
            
        loss = (positive_weight * positive_sample_loss) + ((1 - positive_weight) * negative_sample_loss)

        loss_record = {
            'Sigmoid_hard_positive_sample_loss': positive_sample_loss.item(),
            'Sigmoid_hard_negative_sample_loss': negative_sample_loss.item(),
            'Sigmoid_hard_loss': loss.item(),
        }
        if self.args.learn_hard_positive_weight:
            loss_record['learned_weight'] = positive_weight.item()
            
        return loss, loss_record


class CrossEntropyLoss(nn.Module):
    def __init__(self, args):
        super(CrossEntropyLoss, self).__init__()
        self.args = args
        label_smoothing = getattr(args, 'label_smoothing', 0.0)
        
        if self.args.subsampling:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, similarity, subsampling_weight=None):

        batch_size = similarity.size(0) 
        labels = torch.zeros(batch_size, dtype=torch.long).to(similarity.device)
        loss_per_sample = self.loss_fn(similarity, labels)
        

        if self.args.subsampling and subsampling_weight is not None:

            loss = (subsampling_weight * loss_per_sample).sum() / subsampling_weight.sum()
        else:
            loss = loss_per_sample
        
        # 5. 记录损失值
        loss_record = {
            'CrossEntropy_loss': loss.item()
        }

        return loss, loss_record


class KL_divergency(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temperature = self.args.distill_temperature
        self.stu_preprocess = args.stu_distill_pre_process
        self.tea_preprocess = args.tea_distill_pre_process
        self.spe_temperature = 0.1
        if args.dataset == 'WN18RR':
            self.Euc_nor_temprature = 1.0
        if args.dataset == 'FB15k-237':
            self.Euc_nor_temprature = 0.6
    
    def local_standardize(self, scores, eps=1e-6):
        scores_mean = scores.mean(dim=-1, keepdim=True)
        scores_sqrtvar = torch.sqrt(scores.var(dim=-1, keepdim=True) + eps)
        scores_norm = (scores - scores_mean) / scores_sqrtvar
        return scores_norm, scores_mean, scores_sqrtvar
    
    def pre_process(self, stu_dis, tea_dis):
        if self.stu_preprocess == 'local_standardize':
            stu_dis, _, _ = self.local_standardize(stu_dis)
            
        if (self.tea_preprocess == 'local_standardize') :
            tea_dis, _, _ = self.local_standardize(tea_dis)
        
        return stu_dis, tea_dis
    
    def forward(self, student_dis, teacher_dis, losstype='HyperTeacher', reduction='batchmean'):
        if losstype == 'HyperTeacher':
            student_dis, teacher_dis = self.pre_process(student_dis, teacher_dis)
            teacher_p = F.softmax(teacher_dis / self.temperature, dim=-1).detach()
            student_log_p = F.log_softmax(student_dis / self.temperature, dim=-1)
            
            loss = F.kl_div(student_log_p, teacher_p, reduction=reduction) * (self.temperature ** 2)
        elif losstype == 'optimalTeacher':
            student_dis, teacher_dis = self.pre_process(student_dis, teacher_dis)
            teacher_p = F.softmax(teacher_dis / self.spe_temperature, dim=-1).detach()
            student_log_p = F.log_softmax(student_dis / self.spe_temperature, dim=-1)
            loss = F.kl_div(student_log_p, teacher_p, reduction=reduction) * (self.spe_temperature ** 2)
            
        elif losstype == 'EucTeacher':
            student_dis, teacher_dis = self.pre_process(student_dis, teacher_dis)
            teacher_p = F.softmax(teacher_dis / self.Euc_nor_temprature, dim=-1).detach()
            student_log_p = F.log_softmax(student_dis / self.Euc_nor_temprature, dim=-1)
            loss = F.kl_div(student_log_p, teacher_p, reduction=reduction) * (self.Euc_nor_temprature ** 2)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.tau  = self.args.contrastive_tau
        self.teacher_Euc_embedding_dim = self.args.feature_teacher_Euc_embedding_dim
        self.teacher_Hyper_embedding_dim = self.args.feature_teacher_Hyper_embedding_dim
        self.student_entity_embedding_dim = self.args.target_entity_dim
        self.student_relation_embedding_dim = self.args.target_relation_dim
        self.hidden_dim = self.args.CL_hidden_dim
        self.layermul = self.args.CL_layermul

        self.Teacher1MLP = nn.Sequential(
            nn.Linear(self.teacher_Euc_embedding_dim*2, self.hidden_dim * self.layermul),
            nn.GELU(),
            nn.Linear(self.hidden_dim * self.layermul, self.student_entity_embedding_dim),
            nn.LayerNorm(self.student_entity_embedding_dim)
        )

        self.Teacher2MLP = nn.Sequential(
            nn.Linear(self.teacher_Hyper_embedding_dim*2, self.hidden_dim * self.layermul),
            nn.GELU(),
            nn.Linear(self.hidden_dim * self.layermul, self.student_entity_embedding_dim),
            nn.LayerNorm(self.student_entity_embedding_dim)
        )


    def contrastive_similarity(self, stu_embedding, tea_embedding):
        # Remove the singleton dimension
        stu_embedding = stu_embedding.squeeze(1)  # shape: [batch, embedding_dim]
        tea_embedding = tea_embedding.squeeze(1)  # shape: [batch, embedding_dim]

        # Normalize embeddings
        stu_embedding = torch.nn.functional.normalize(stu_embedding, p=2, dim=1)
        tea_embedding = torch.nn.functional.normalize(tea_embedding, p=2, dim=1)

        cosine_similarity_matrix = torch.matmul(stu_embedding, tea_embedding.T)
        cosine_similarity_matrix = cosine_similarity_matrix / self.tau
        softmax_score = F.log_softmax(cosine_similarity_matrix, dim=1)
        labels = torch.arange(cosine_similarity_matrix.size(0)).to(stu_embedding.device)  # shape: [batch]
        loss = F.nll_loss(softmax_score, labels)

        return loss


    def forward(self, eh, PT_head1, PT_head2, weight=None):
        stu_head = eh
        
        tea_head1 = self.Teacher1MLP(PT_head1)
        tea_head2 = self.Teacher2MLP(PT_head2)
        
        if weight is not None:
            weight = weight.squeeze(1)
            w1 = weight[:, 0:1].unsqueeze(-1)        # [B, 1, 1]
            w2 = weight[:, 1:2].unsqueeze(-1)        # [B, 1, 1]
            combined_tea_head = w1 * tea_head1 + w2 * tea_head2
        else:
            combined_tea_head = tea_head1 + tea_head2

        head_loss = self.contrastive_similarity(stu_head, combined_tea_head)
        loss = head_loss

        return loss


class Total_Loss(nn.Module):
    def __init__(self, args):
        super(Total_Loss, self).__init__()
        self.args = args
        self.distribution_selector = DistributionSelector(args)
        self.weight_learner = weight_learner(args)
        self.prior_weight_learner = Prior_Weight(args)
        self.weight_learner_tau = args.weight_learner_tau
        if args.hard_loss_function == 'SigmoidLoss':
            self.stu_hard_loss = SigmoidLoss(args, 'stu')
        elif args.hard_loss_function == "CrossEntropy":
            self.stu_hard_loss = CrossEntropyLoss(args)
        self.tea_hard_loss = SigmoidLoss(args, 'tea')
        self.soft_loss = KL_divergency(args)
        self.CL_loss = ContrastiveLoss(args)
    
    def get_weights(self, head, relation, tail, stu_score, Euc_tea_score, Hyper_tea_score, data):
        x = self.weight_learner(head, relation, tail, stu_score, Euc_tea_score, Hyper_tea_score, data)
        if self.args.final_function == 'softmax':
            weights = F.softmax(x / self.weight_learner_tau, dim=2)
        elif self.args.final_function == 'gumbel_softmax':
            weights = F.gumbel_softmax(x, tau=self.weight_learner_tau, hard=True, dim=-1)
        

        prior_weight = self.prior_weight_learner(data)
        weights = (weights + 1.0 * prior_weight)/2
        
        return weights
    
    def get_optimal_tea_score(self, Euc_tea_score, Hyper_tea_score, eps=1e-8):
        target_mean, target_std = self.distribution_selector(Euc_tea_score, Hyper_tea_score)
        target_std = target_std + eps
        
        Euc_mean = torch.mean(Euc_tea_score, dim=1, keepdim=True)
        Euc_std = torch.std(Euc_tea_score, dim=1, keepdim=True) + eps

        Hyper_mean = torch.mean(Hyper_tea_score, dim=1, keepdim=True)
        Hyper_std = torch.std(Hyper_tea_score, dim=1, keepdim=True) + eps

        Euc_map_score = (Euc_tea_score - Euc_mean) / Euc_std 
        Euc_map_score = Euc_map_score * target_std + target_mean 

        Hyper_map_score = (Hyper_tea_score - Hyper_mean) / Hyper_std
        Hyper_map_score = Hyper_map_score * target_std + target_mean
        
        return Euc_map_score, Hyper_map_score
    
    def get_optimal_score(self, Euc_score, Hyper_score, weights):
        
        optimal_tea_score = weights[..., 0] * Euc_score + weights[..., 1] * Hyper_score
        
        return optimal_tea_score
    
    
    def dynamic_loss_weight(self, epoch):
        start_Euc_weight = self.args.start_Euc_weight
        start_Hyper_weight = self.args.start_Hyper_weight
        start_Opt_weight = self.args.start_Opt_weight

        end_Euc_weight = self.args.end_Euc_weight
        end_Hyper_weight = self.args.end_Hyper_weight
        end_Opt_weight = self.args.end_Opt_weight
        
        stop_epoch = self.args.stop_epoch
        
        total_epochs_for_transition = min(stop_epoch, self.args.epoch)

        raw_alpha = (epoch - 1) / total_epochs_for_transition
        alpha = max(0.0, min(1.0, raw_alpha))

        Euc_weight = start_Euc_weight + (end_Euc_weight - start_Euc_weight) * alpha
        Hyper_weight = start_Hyper_weight + (end_Hyper_weight - start_Hyper_weight) * alpha 
        Opt_weight = start_Opt_weight + (end_Opt_weight - start_Opt_weight) * alpha
        
        if epoch > stop_epoch:
            Euc_weight, Hyper_weight, Opt_weight = end_Euc_weight, end_Hyper_weight, end_Opt_weight
        
        return Euc_weight, Hyper_weight, Opt_weight

    
    
    def forward(self, stu_score, stu_embeddings, Euc_tea_score, Euc_embeddings, Hyper_tea_score, Hyper_embeddings, data, epoch):
        head, relation, tail = stu_embeddings
        Euc_tea_head, _, _ = Euc_embeddings
        Hyper_tea_head, _, _ = Hyper_embeddings
        

        Euc_map_score, Hyper_map_score = self.get_optimal_tea_score(Euc_tea_score, Hyper_tea_score)
        
        weights = self.get_weights(head, relation, tail, stu_score, Euc_map_score, Hyper_map_score, data)
        optimal_tea_score = self.get_optimal_score(Euc_map_score, Hyper_map_score, weights)
                
        hard_loss, hard_loss_record = self.stu_hard_loss(stu_score)
        
        Euc_soft_loss = self.soft_loss(stu_score, Euc_tea_score, losstype="EucTeacher")
        Hyper_soft_loss = self.soft_loss(stu_score, Hyper_tea_score, losstype="HyperTeacher")
        Optimal_soft_loss = self.soft_loss(stu_score, optimal_tea_score, losstype="optimalTeacher")
        Optimal_tea_loss, _ = self.tea_hard_loss(optimal_tea_score)
        contrasitve_loss = self.CL_loss(head, Euc_tea_head, Hyper_tea_head, weights)
        
        if not self.args.use_dynamic_loss_weight:
            total_loss = self.args.hard_loss_weight * hard_loss + \
                    self.args.euc_soft_loss_weight * Euc_soft_loss + \
                    self.args.hyper_soft_loss_weight * Hyper_soft_loss + \
                    self.args.optimal_soft_loss_weight * Optimal_soft_loss + \
                    self.args.optimal_tea_loss_weight * Optimal_tea_loss + \
                    self.args.contrastive_loss_weight * contrasitve_loss
        else:
            Euc_weight, Hyper_weight, Opt_weight = self.dynamic_loss_weight(epoch)
            total_loss = self.args.hard_loss_weight * hard_loss + \
                    Euc_weight * Euc_soft_loss + \
                    Hyper_weight * Hyper_soft_loss + \
                    Opt_weight * Optimal_soft_loss + \
                    self.args.optimal_tea_loss_weight * Optimal_tea_loss + \
                    self.args.contrastive_loss_weight * contrasitve_loss
        
        loss_record = hard_loss_record
        other_loss_record = {
            'total_loss': total_loss.item()
        }
        total_loss_record = {**loss_record, **other_loss_record}

        return total_loss, total_loss_record
        