#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import torch
import time
import random
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TrainDataset(Dataset):
    def __init__(self, triples, args):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.args = args
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.negative_sample_size = args.negative_sample_size
        self.count = self.count_frequency(triples)
        self.true_triples = self.get_true_head_and_tail(self.triples)
        self.query_aware_dict1, self.query_aware_dict2 = self.read_qad(args.qt_dict_path)
        self.relation_aware_dict = self.read_rad(args.rt_dict_path)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.count[(head, relation)]
        
        if relation >= self.nrelation/2: 
            subsampling_weight += self.count[(tail, relation-int(self.nrelation/2))]
        else: 
            subsampling_weight += self.count[(tail, relation+int(self.nrelation/2))]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        if self.args.pre_sample_size == 0:
            negative_sample_list = []
            negative_sample_size = 0
        else:
            pre_sample_list, forbidden_tails = self.pre_sampling(head, relation, tail, pre_sample_num=self.args.pre_sample_size) # 从第二个教师模型采样
            negative_sample_list = [pre_sample_list]
            negative_sample_size = len(pre_sample_list)
        
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.args.pre_sample_size != 0:
               true_tails = self.true_triples[(head, relation)]
               teacher_tails = forbidden_tails 
               exclude_set = np.union1d(true_tails, teacher_tails)
            else:
                exclude_set = self.true_triples[(head, relation)]

            mask = np.in1d(
                negative_sample, 
                exclude_set,
                assume_unique=False, 
                invert=True 
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        np.random.shuffle(negative_sample)
        
        if head not in self.true_triples[(head, relation)]:
            negative_sample[random.randint(0, self.negative_sample_size - 1)] = head

        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        
        return positive_sample, negative_sample, subsampling_weight, 'QueryAwareSample'
    
    def read_rad(self, path):
        result = {}
        try:
            with open(path, 'r') as file:
                for line in file:
                    parts = line.strip().split()  
                    if parts:  
                        key = int(parts[0])
                        values = list(map(int, parts[1:]))
                        result[key] = np.array(np.unique(values))
        except FileNotFoundError:
            return {}
        
        return result
    
    def read_qad(self, path):
        PT1_result = {}
        PT2_result = {}
        try:
            with open(path, 'r') as file:
                lines = file.readlines()
                for i in range(0, len(lines), 3):
                    head_relation = lines[i].strip().split("\t")
                    head, relation = int(head_relation[0]), int(head_relation[1])

                    pt1_ids = list(map(int, lines[i + 1].strip().split("\t")))
                    pt2_ids = list(map(int, lines[i + 2].strip().split("\t")))

                    # PT1_result[(head, relation)] = np.array(np.unique(pt1_ids))
                    # PT2_result[(head, relation)] = np.array(np.unique(pt2_ids))
                    
                    PT1_result[(head, relation)] = np.array(pt1_ids)
                    PT2_result[(head, relation)] = np.array(pt2_ids)
                    
        except FileNotFoundError:
            return {}, {}
        
        return PT1_result, PT2_result
    
    def relation_pre_sampling(self, head, relation, tail, RAS=30, invalid_ids=None):
        relation_aware_tail = self.relation_aware_dict[relation]
        mask = np.in1d(
            relation_aware_tail, 
            np.intersect1d(invalid_ids, self.true_triples[(head, relation)], assume_unique=False),
            assume_unique=False, 
            invert=True 
        ) 
        relation_aware_tail = relation_aware_tail[mask]
        np.random.shuffle(relation_aware_tail)
        
        selected_tail = relation_aware_tail[:RAS]
        return selected_tail
    
    def pre_sampling(self, head, relation, tail, pre_sample_num=50):
        query_aware_tail = self.query_aware_dict2[(head, relation)]
        if self.args.filter_true_triples: 
            mask = np.in1d(
                query_aware_tail, 
                np.append(self.true_triples[(head, relation)], head),
                assume_unique=False, 
                invert=True 
            ) 
            query_aware_tail = query_aware_tail[mask]
            
        if not self.args.pre_sample_top: 
            np.random.shuffle(query_aware_tail)
            forbidden_tails = query_aware_tail
        else:
            forbidden_tails = query_aware_tail[:pre_sample_num]
        
        selected_tail = query_aware_tail[:pre_sample_num]
        
        return selected_tail, forbidden_tails
    
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_triples = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_triples:
                true_triples[(head, relation)] = []
            true_triples[(head, relation)].append(tail)

        for head, relation in true_triples:
            true_triples[(head, relation)] = np.array(list(set(true_triples[(head, relation)])))             

        return true_triples
    

class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, args):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.args = args

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.args.only_get_qtdict:
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (0, rand_tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                    else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)

        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        
        return positive_sample, negative_sample, filter_bias





class SimpleBidirectionalOneShotIterator(object):
    def __init__(self, train_dataloader):
        self.train_dataloader = self.one_shot_iterator(train_dataloader)
        self.total_len = len(train_dataloader)
        
    def __next__(self):
        data = next(self.train_dataloader)

        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data