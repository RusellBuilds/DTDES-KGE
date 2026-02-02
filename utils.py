# utils.py

import argparse
import json
import logging
import os
import random
import copy
import re
import numpy as np
import torch


def save_args(args, file_path):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    args_dict = vars(args)
    with open(file_path, 'w') as f:

        json.dump(args_dict, f, indent=4, sort_keys=True)



def save_model(model,args):
    save_path = os.path.join(args.save_path, 'checkpoint')
    save_dict = {
        'model_state_dict': model.state_dict(),
        'args': args
    }
    torch.save(save_dict, save_path)
    logging.info(f"Student model checkpoint saved to {save_path}")



def set_logger(args):
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, 'train.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, epoch, metrics):
    for metric in metrics:
        logging.info('%s %s at epoch %d: %f' % (mode, metric, epoch, metrics[metric]))

def calculate_metrics(logs, epoch):
    if not logs:
        return
    metrics = {}
    metric_counts = {}
    for log in logs:
        for metric, value in log.items():
            if metric not in metrics:
                metrics[metric] = 0.0
                metric_counts[metric] = 0
            metrics[metric] += value
            metric_counts[metric] += 1
    for metric in metrics.keys():
        if metric_counts[metric] > 0:
            metrics[metric] /= metric_counts[metric]
    log_metrics('Train', epoch, metrics)


def get_query_tail_dict(train_data_path, relation_num): 
    train_true_triples = {}
    with open(train_data_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            h, r, t = int(h), int(r), int(t)
            if (h,r) not in train_true_triples:
                train_true_triples[(h,r)] = []
            train_true_triples[(h,r)].append(t)
            
            if(t, r+relation_num) not in train_true_triples:
                train_true_triples[(t,r+relation_num)] = []
            train_true_triples[(t,r+relation_num)].append(h)
    
    return train_true_triples

def read_tripels_with_ids(file_path): 
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            h, r, t = int(h), int(r), int(t)
            triples.append((h,r,t))
    return triples

def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def read_triple_with_reverse(file_path, entity2id, relation2id):
    triples = []
    relation_num = len(relation2id)
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
            triples.append((entity2id[t], relation2id[r] + relation_num, entity2id[h]))
    return triples

def read_data_reverse(args):
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = {line.strip().split('\t')[1]: int(line.strip().split('\t')[0]) for line in fin}
    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = {line.strip().split('\t')[1]: int(line.strip().split('\t')[0]) for line in fin}
    
    nentity = len(entity2id)
    nrelation = len(relation2id) * 2
    
    train_triples = read_triple_with_reverse(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple_with_reverse(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple_with_reverse(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    logging.info('#total entity: %d' % nentity)
    logging.info('#total relation: %d' % nrelation)
    
    all_true_triples = train_triples + valid_triples + test_triples
    
    return train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation

def parse_args_from_json(json_path):
    args = argparse.Namespace()
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 配置文件未找到: {json_path}")
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
        
    def dict_to_namespace(d, namespace):
        for key, value in d.items():
            if key == 'hyperparameter_tuning':
                continue
            if isinstance(value, dict):
                dict_to_namespace(value, namespace)
            else:
                setattr(namespace, key, value)

    dict_to_namespace(config_dict, args)
    return args

def override_args_from_dict(args, override_dict):
    for key, value in override_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            setattr(args, key, value)
    return args