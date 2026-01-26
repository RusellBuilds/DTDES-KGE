import argparse
import json
import logging
import os
import random
import copy
import torch
import tqdm
import itertools
import sys
import re

import numpy as np
from torch.utils.data import DataLoader

from utils import (
    read_data_reverse, calculate_metrics, set_logger, log_metrics, 
    parse_args_from_json, override_args_from_dict, save_args, save_model,
    get_query_tail_dict, read_tripels_with_ids
)
from code.dataloader.dataloader import TrainDataset, TestDataset, SimpleBidirectionalOneShotIterator
from code.KGE.LorentzKG_Reverse.LorentzKG_Reverse import HyperNet
from code.KGE.HAKE_Reverse import HAKE_Reverse
from code.KGE.RotatE_Reverse import RotatE_Reverse
from code.KGE.SCCF_Reverse import SCCF_Reverse
from code.Loss.loss import Total_Loss



def main(args):
    save_args_path = os.path.join(args.save_path, 'config_run.json')
    save_args(args, save_args_path)
    # ----------------------------------------

    if args.cuda and torch.cuda.is_available() and hasattr(args, 'gpu_id') and args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)
    
    train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation = read_data_reverse(args)
    
    if args.only_get_qtdict:
        if 'FB15k-237' in args.data_path:
            test_triples = read_tripels_with_ids('data/FB15k-237/single_query.txt')
        elif 'WN18RR' in args.data_path:  
            test_triples = read_tripels_with_ids('data/WN18RR/single_query.txt')
        elif 'YAGO3-10' in args.data_path:
            test_triples = read_tripels_with_ids('data/YAGO3-10/single_query.txt')
            
    args.nentity = nentity
    args.nrelation = nrelation
    
    train_dataloader = DataLoader(
        TrainDataset(train_triples, args), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )
    train_dataloader = SimpleBidirectionalOneShotIterator(train_dataloader)
    
    valid_dataloader= DataLoader(
        TestDataset(valid_triples, all_true_triples, args.nentity, args.nrelation, args), 
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num//2), 
        collate_fn=TestDataset.collate_fn
    )
    valid_dataloader = [valid_dataloader]
    
    test_dataloader = DataLoader(
        TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, args), 
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num//2), 
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader = [test_dataloader]

    if args.dataset == 'FB15k-237':
        logits_Hyper_teacher = HyperNet(dims=args.logits_teacher_Hyper_dim, margin=args.logits_teacher_Hyper_margin, noise_reg=args.logits_teacher_Hyper_noise, npos=args.logits_teacher_Hyper_npos, max_norm=args.logits_teacher_Hyper_maxnorm, args=args)
        feature_Euc_teacher = RotatE_Reverse(args, 'Euc')
        feature_Hyper_teacher = RotatE_Reverse(args, 'Hyper')
    elif args.dataset == 'WN18RR':
        logits_Hyper_teacher = HyperNet(dims=args.logits_teacher_Hyper_dim, margin=args.logits_teacher_Hyper_margin, noise_reg=args.logits_teacher_Hyper_noise, npos=args.logits_teacher_Hyper_npos, max_norm=args.logits_teacher_Hyper_maxnorm, args=args)
        logits_Euc_teacher = HAKE_Reverse(args, 'Euc')
        feature_Euc_teacher = RotatE_Reverse(args, 'Euc')
        feature_Hyper_teacher = RotatE_Reverse(args, 'Hyper')    
    else:
        raise ValueError(f"Unknown args.dataset: {args.dataset}")

    if not args.execute_ditill_teacher_to_rotate:
        student = SCCF_Reverse(args)
    else:
        student = RotatE_Reverse(args, 'Stu') 
    Loss = Total_Loss(args)

    if args.cuda:
       student.cuda()
       Loss.cuda()
       logits_Hyper_teacher.cuda()
       feature_Euc_teacher.cuda()
       feature_Hyper_teacher.cuda() 
       if args.dataset == 'WN18RR':
           logits_Euc_teacher.cuda()
    
    student_params = student.parameters()
    loss_params = Loss.parameters()
    all_learnable_params = itertools.chain(student_params, loss_params)
    milestones = []
    start = args.milestone
    while start < args.epoch:
        milestones.append(start)
        start = start * 2
    
    optimizer = torch.optim.Adam(all_learnable_params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.decreasing_lr)

    training_loss = []
    train_data_len = train_dataloader.total_len
    
    if args.only_test or args.only_Euc_teacher_test or args.only_Hyper_teacher_test or args.only_get_qtdict:
        metrics = test_model(args, test_dataloader, student, logits_Euc_teacher, logits_Hyper_teacher)
        log_metrics('Test', args.epoch, metrics)
        sys.exit(0)
    
    for epoch in range(1, args.epoch + 1):
        student.train()
        Loss.train()
        with tqdm.tqdm(total=train_data_len, initial=0) as bar:
            bar.set_description(f'Epoch {epoch} Train Loss')
            for step in range(1, train_data_len + 1):
                optimizer.zero_grad()
                positive_sample, negative_sample, subsampling_weight, mode = next(train_dataloader)
                if args.cuda:
                    positive_sample, negative_sample, subsampling_weight = positive_sample.cuda(), negative_sample.cuda(), subsampling_weight.cuda()
                
                if args.dataset == 'WN18RR':
                    Euc_tea_score = logits_Euc_teacher((positive_sample, negative_sample))
                elif args.dataset == 'FB15k-237':
                    Euc_tea_score = feature_Euc_teacher((positive_sample, negative_sample))
                Euc_tea_embeddings = feature_Euc_teacher.get_embedding((positive_sample, negative_sample))
                Hyper_tea_score = logits_Hyper_teacher((positive_sample, negative_sample))
                Hyper_tea_embeddings = feature_Hyper_teacher.get_embedding((positive_sample, negative_sample))
                stu_score = student((positive_sample, negative_sample))
                stu_embeddings = student.get_embedding((positive_sample, negative_sample))

                loss, loss_record = Loss(stu_score, stu_embeddings, Euc_tea_score, Euc_tea_embeddings, Hyper_tea_score, Hyper_tea_embeddings, (positive_sample, negative_sample), epoch)

                if torch.isnan(loss):
                    raise ValueError("NaN loss detected") 
                
                loss.backward()
                optimizer.step()
                
                training_loss.append(loss_record)
                bar.update(1)
                if 'total_loss' in loss_record:
                    bar.set_postfix(loss=loss_record['total_loss'])
            
            calculate_metrics(training_loss, epoch)
            training_loss = []
        
        scheduler.step()
        
        if (epoch % args.save_checkpoint) == 0:
            save_model(student,args)
        if (epoch % args.test_per_epochs) == 0:
            metrics = test_model(args, valid_dataloader, student)
            log_metrics('Valid', epoch, metrics)
            metrics = test_model(args, test_dataloader, student)
            log_metrics('Test', epoch, metrics)
            if epoch == args.epoch:
                return 1
                
    if args.epoch % args.test_per_epochs != 0:
        metrics = test_model(args, valid_dataloader, student)
        log_metrics('Valid', epoch, metrics)
        metrics = test_model(args, test_dataloader, student)
        log_metrics('Test', args.epoch, metrics)
    
    return 1


def test_model(args, testdataloader, student, Euc_teacher=None, Hyper_teacher=None):
    if args.only_get_qtdict:
        if 'FB15k-237' in args.data_path:
            train_true_triples = get_query_tail_dict(train_data_path='data/FB15k-237/train', relation_num=237)
        elif 'WN18RR' in args.data_path:
            train_true_triples = get_query_tail_dict(train_data_path='data/WN18RR/train', relation_num=11)

    
    student.eval()
    with torch.no_grad():
        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in testdataloader])
        result_record = []
        with tqdm.tqdm(total=total_steps, unit='ex') as bar:
            bar.set_description(f'Evaluation')
            for test_dataset in testdataloader:
                for positive_sample, negative_sample, filter_bias in test_dataset:
                    if args.cuda:
                        positive_sample, negative_sample, filter_bias = positive_sample.cuda(), negative_sample.cuda(), filter_bias.cuda()
                        
                    batch_size = positive_sample.size(0)
                    
                    if args.only_get_qtdict:
                        Euc_score = Euc_teacher((positive_sample, negative_sample))
                        Hyper_score = Hyper_teacher((positive_sample, negative_sample))
                        score = Hyper_score
                        get_querytail_dict(Euc_score[:, 1:], Hyper_score[:, 1:], positive_sample, args, train_true_triples, 100)
                        
                    else:
                        if args.only_Euc_teacher_test:
                            score = Euc_teacher((positive_sample, negative_sample))
                        elif args.only_Hyper_teacher_test:
                            score = Hyper_teacher((positive_sample, negative_sample))
                        else:
                            score = student((positive_sample, negative_sample))    
                    
                    score = score[:, 1:]
                    score += filter_bias
                    
                    argsort = torch.argsort(score, dim=1, descending=True)
                    positive_arg = positive_sample[:, 2]

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero(as_tuple=False)
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()
                        
                        logs.append({
                            'MRR': 1.0/ranking, 'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })
                        result_record.append(f"{positive_sample[i, 0].item()}\t{positive_sample[i, 1].item()}\t{positive_sample[i, 2].item()}\t{ranking}")
                    
                    step += len(positive_sample)
                    # bar.update(len(positive_sample))
                    bar.update(1)
        
    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, 'test_detail_result.txt')
    with open(output_path, 'w') as f:
        for record in result_record:
            f.write(record + "\n")
    
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    student.train()
    return metrics



def get_querytail_dict(PT1_score, PT2_score, positive_sample, args, train_true_triples, top_k=100):
    batch_size = PT1_score.shape[0]
    with open(args.qt_dict_name, 'a') as file:
        for batch in range(batch_size):
            PT1 = PT1_score[batch]
            PT2 = PT2_score[batch]
            head, relation, tail = positive_sample[batch]

            invalid_entities = set(train_true_triples.get((head.item(), relation.item()), []))

            # 转换为 tensor
            invalid_array = torch.tensor(list(invalid_entities), device=PT1.device)
            
            # 自定义 isin 功能
            def isin_check(indices, invalid_array):
                mask = (indices.unsqueeze(1) == invalid_array).any(dim=1)
                return ~mask

            PT1_sorted_indices = PT1.argsort(descending=True)
            if args.perform_filtering: 
                valid_mask_PT1 = isin_check(PT1_sorted_indices, invalid_array)
            else:
                valid_mask_PT1 = torch.ones_like(PT1_sorted_indices, dtype=torch.bool)
            PT1_filtered = PT1_sorted_indices[valid_mask_PT1][:top_k].tolist()

            PT2_sorted_indices = PT2.argsort(descending=True)
            if args.perform_filtering:
                valid_mask_PT2 = isin_check(PT2_sorted_indices, invalid_array)
            else:
                valid_mask_PT2 = torch.ones_like(PT2_sorted_indices, dtype=torch.bool)
            PT2_filtered = PT2_sorted_indices[valid_mask_PT2][:top_k].tolist()

            file.write(f"{head}\t{relation}\n")
            file.write("\t".join(map(str, PT1_filtered)) + "\n")
            file.write("\t".join(map(str, PT2_filtered)) + "\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('config', type=str, nargs='?', default='config/YAGO.json',
                       )
    
    initial_args = parser.parse_args()
    config_file_path = initial_args.config
    
    with open(config_file_path, 'r') as f:
        config = json.load(f)
        
    base_args = parse_args_from_json(config_file_path)
    hyperparameter_space = config.get('hyperparameter_tuning', {})
    
    MAX_RETRIES = 3
    
    if 'description' in hyperparameter_space:
        del hyperparameter_space['description']

    if not hyperparameter_space:
        paths_to_format = ['save_path', 'qt_dict_path', 'rt_dict_path']
        for path_key in paths_to_format:
            if hasattr(base_args, path_key):
                original_path = getattr(base_args, path_key)
                if isinstance(original_path, str) and '{' in original_path:
                    setattr(base_args, path_key, original_path.format(**vars(base_args).get('experiment_setup', vars(base_args))))
        set_logger(base_args)
        main(base_args)
    else:
        
        param_names = list(hyperparameter_space.keys())
        param_values = list(hyperparameter_space.values())
        all_combinations = list(itertools.product(*param_values))
        total_runs = len(all_combinations)
        
        
        initial_run_id = int(base_args.run_id) if hasattr(base_args, 'run_id') and base_args.run_id else 1000
        
        for i, combo in enumerate(all_combinations):
            
            combo_succeeded = False
            for attempt in range(MAX_RETRIES):
                current_args = copy.deepcopy(base_args)
                override_dict = dict(zip(param_names, combo))
                current_run_id = initial_run_id + i
                override_dict['run_id'] = current_run_id
                current_args = override_args_from_dict(current_args, override_dict)
                
                paths_to_format = ['save_path', 'qt_dict_path', 'rt_dict_path']
                args_dict = vars(current_args)
                flat_args_dict = {}
                for k, v in args_dict.items():
                    if isinstance(v, argparse.Namespace):
                        flat_args_dict.update(vars(v))
                    else:
                        flat_args_dict[k] = v

                for path_key in paths_to_format:
                    if hasattr(current_args, path_key):
                        original_path = getattr(current_args, path_key)
                        if isinstance(original_path, str) and '{' in original_path:
                            setattr(current_args, path_key, original_path.format(**flat_args_dict))

                for param, value in override_dict.items():
                    print(f"    - {param}: {value}")
                print("="*80 + "\n")

                set_logger(current_args)

                try:
                    if main(current_args):
                        combo_succeeded = True
                        break 

                except ValueError as e:
                    if "NaN loss detected" in str(e):
                        pass
                    else:
                        break 
                except Exception as e:
                    pass

            if not combo_succeeded:
                pass

        print("\nOver。")