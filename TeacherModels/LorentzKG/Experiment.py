from collections import defaultdict
from copy import deepcopy
import os
import sys
import logging
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from load_data import Data
from LorentzModel import HyperNet
from optim.radam import RiemannianAdam
from optim.rsgd import RiemannianSGD

class KGETrainingDataset(Dataset):
    """
    一个自定义的 PyTorch Dataset 类，用于知识图谱嵌入训练。
    它负责处理单个正样本的获取，以及为其生成负样本（包括“真负采样”）。
    """
    def __init__(self, data_idxs_np, num_entities, nneg, sr_vocab, real_neg):
        self.data_idxs_np = data_idxs_np
        self.num_entities = num_entities
        self.nneg = nneg
        self.sr_vocab = sr_vocab
        self.real_neg = real_neg

    def __len__(self):
        return len(self.data_idxs_np)

    def __getitem__(self, idx):
        positive_sample = self.data_idxs_np[idx]
        head, relation, tail = positive_sample

        negative_samples = np.random.randint(
            low=0, high=self.num_entities, size=self.nneg
        )

        if self.real_neg:
            filt = self.sr_vocab.get((head, relation), set())
            filt.add(tail)  
            for i in range(self.nneg):
                while negative_samples[i] in filt:
                    negative_samples[i] = np.random.randint(low=0, high=self.num_entities)
        
        return torch.from_numpy(positive_sample), torch.from_numpy(negative_samples)


def set_logger(args):
    folder_name = f"{args.dataset}_{args.run_id}"
    full_path = os.path.join("models", folder_name)
    log_file = os.path.join(full_path, 'train.log')

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


class Experiment:
    def __init__(self,
                 args = None,
                 data = None,
                 margin = 0.5,
                 noise_reg = 0.15,
                 learning_rate=1e-3,
                 dim=40,
                 nneg=50,
                 npos=10,
                 valid_steps=10,
                 num_epochs=500,
                 batch_size=128,
                 max_norm=0.5,
                 max_grad_norm=1,
                 optimizer='radam',
                 cuda=True,
                 early_stop=10,
                 real_neg=True,
                 device='cuda:0',
                 step_size=30,
                 gamma=0.6,
                 run_id=666
                 ):
        self.args = args
        self.data = data
        self.learning_rate = learning_rate
        self.dim = dim
        self.npos = npos
        self.nneg = nneg
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        self.valid_steps = valid_steps
        self.cuda = cuda
        self.early_stop = early_stop
        self.real_neg = real_neg
        self.device = device
        self.margin = margin
        self.noise_reg = noise_reg
        self.step_size = step_size
        self.gamma = gamma
        self.entity_idxs = data.entity2id
        self.relation_idxs = data.relation2id
        
        self.id2entity = {v: k for k, v in self.entity_idxs.items()}
        self.id2relation = {v: k for k, v in self.relation_idxs.items()}
        
        set_logger(self.args)

    def get_data_idxs(self, data):
        data_idxs = [
            (self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
             self.entity_idxs[data[i][2]]) for i in range(len(data))
        ]
        return data_idxs

    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        er_vocab = defaultdict(set)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].add(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, data, batch=100, save_path=None):
        d = self.data
        hits = []
        ranks = []
        rank_by_rela = {}
        hit_by_rela = {}
        for i in range(10):
            hits.append([])

        test_data_idxs = np.array(self.get_data_idxs(data))
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        tt = torch.Tensor(np.array(range(len(d.entities)),
                                  dtype=np.int64)).cuda().long().repeat(
            batch, 1) if self.cuda else torch.Tensor(np.array(range(len(d.entities)),
                                                              dtype=np.int64)).long().repeat(batch, 1)

        f_out = None
        if save_path:
            f_out = open(save_path, 'w', encoding='utf-8')

        try:
            for i in tqdm(range(0, len(test_data_idxs), batch), desc="Evaluating"):
                data_point = test_data_idxs[i:i + batch]
                e1_idx = torch.tensor(data_point[:, 0])
                r_idx = torch.tensor(data_point[:, 1])
                e2_idx = torch.tensor(data_point[:, 2])
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()

                # head batch
                predictions_s_h = model.forward( e1_idx, r_idx, tt[:min(batch, len(test_data_idxs) - i)])
                # tail batch
                predictions_s_t = model.forward(tt[:min(batch, len(test_data_idxs) - i)], torch.where(r_idx>=(len(self.relation_idxs)//2), r_idx-(len(self.relation_idxs)//2), r_idx+(len(self.relation_idxs)//2)), e1_idx )
                predictions_s = torch.stack([predictions_s_t, predictions_s_h], dim=-1)
                predictions_s = torch.mean(predictions_s, dim=-1)

                for j in range(min(batch, len(test_data_idxs) - i)):
                    filt = list(sr_vocab[(data_point[j][0], data_point[j][1])])
                    target_value = predictions_s[j][e2_idx[j]].item()
                    predictions_s[j][filt] = -np.Inf
                    predictions_s[j][e1_idx[j]] = -np.Inf
                    predictions_s[j][e2_idx[j]] = target_value

                    rank = (predictions_s[j] >= target_value).sum().item() - 1

                    if f_out:
                        head_str = self.id2entity[data_point[j][0]]
                        rel_str = self.id2relation[data_point[j][1]]
                        tail_str = self.id2entity[data_point[j][2]]
                        f_out.write(f"{head_str}\t{rel_str}\t{tail_str}\t{rank + 1}\n")
                    
                    current_rank = rank + 1
                    ranks.append(current_rank)
                    rank_by_rela.setdefault(data_point[j][1]-1 if data_point[j][1]%2==1 else data_point[j][1],
                                            []).append(current_rank)

                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                            hit_by_rela.setdefault(
                                data_point[j][1]-1 if data_point[j][1]%2==1 else data_point[j][1],
                                [[] for ii in range(10)])[hits_level].append(1.)
                        else:
                            hits[hits_level].append(0.0)
                            hit_by_rela.setdefault(
                                data_point[j][1]-1 if data_point[j][1]%2==1 else data_point[j][1],
                                [[] for ii in range(10)])[hits_level].append(0.)
        finally:
            if f_out:
                f_out.close()

        return np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(1. / (np.array(ranks)))

    @property
    def train_and_eval(self):
        d = self.data
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        
        model = HyperNet(self.args, d, self.dim, self.max_norm, self.margin, self.nneg, self.npos, self.noise_reg)
        
        print("Training the %s model..." % "HyperNet")
        if self.optimizer == 'radam':
            opt = RiemannianAdam(model.parameters(),
                                 lr=self.learning_rate,
                                 stabilize=1)
        elif self.optimizer == 'rsgd':
            opt = RiemannianSGD(model.parameters(),
                                 lr=self.learning_rate,
                                 stabilize=1)
        elif self.optimizer == 'adam':
            opt = Adam(model.parameters(),
                       lr=self.learning_rate)
        else:
            raise ValueError("Wrong optimizer")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt,
                                                     step_size=self.step_size, gamma=self.gamma,
                                                     verbose=True)

        if self.cuda:
            model.to(self.device)

        # 预先计算一次全局的 (h, r) -> {t} 映射，用于负采样
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        train_data_idxs_np = np.array(train_data_idxs)
        
        train_dataset = KGETrainingDataset(
            train_data_idxs_np,
            len(self.entity_idxs),
            self.nneg,
            sr_vocab,
            self.real_neg
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size * self.npos,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True, 
            drop_last=True  
        )
        
        targets = np.zeros((self.batch_size, self.nneg * self.npos + self.npos))
        targets[:, 0:self.npos] = 1
        targets = torch.FloatTensor(targets).to(self.device)

        max_mrr = 0.0
        max_it = 0
        mrr = 0
        bad_cnt = 0
        print("Starting training...")
        
        folder_name = f"{self.args.dataset}_{self.args.run_id}"
        full_path = os.path.join("models", folder_name)
        evaluation_ranks_file = os.path.join(full_path, 'evaluation_ranks.txt')
        checkpoint_file = os.path.join(full_path, 'best_checkpoint.pth')

        for it in range(1, self.num_epochs + 1):
            model.train()
            losses = []
            
            for j, (positive_batch, negative_batch) in enumerate(tqdm(data_loader, desc=f"Epoch {it}/{self.num_epochs}")):
                
                positive_batch = positive_batch.to(self.device, non_blocking=True)
                negative_batch = negative_batch.to(self.device, non_blocking=True)

                data_batch = positive_batch.view(self.batch_size, self.npos, 3)
                negsamples = negative_batch.view(self.batch_size, self.npos, self.nneg)
                
                opt.zero_grad()
                
                e1_idx = data_batch[:,:, 0]
                r_idx = data_batch[:,:, 1]
                e2_idx = torch.cat([data_batch[:,:, 2:3], negsamples], dim=-1)
                
                intervals = model.forward(e1_idx, r_idx, e2_idx)
                BCELoss = model.BCEloss(intervals,targets)
                CRRLoss = model.CRRloss(-intervals,targets)
                MarginLoss = model.Marginloss(intervals,targets)
                loss = self.args.BCE_w * BCELoss + self.args.CRR_w * CRRLoss + self.args.Margin_w * MarginLoss

                r_idx_rev = torch.where(r_idx>=(len(self.relation_idxs)//2), r_idx-(len(self.relation_idxs)//2), r_idx+(len(self.relation_idxs)//2))
                intervals_rev = model.forward(e2_idx, r_idx_rev, e1_idx)
                BCELoss = model.BCEloss(intervals_rev,targets)
                CRRLoss = model.CRRloss(-intervals_rev,targets)
                MarginLoss = model.Marginloss(intervals,targets)
                loss += self.args.BCE_w * BCELoss + self.args.CRR_w * CRRLoss + self.args.Margin_w * MarginLoss

                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm, error_if_nonfinite=True)
                opt.step()
                losses.append(loss.item())

            scheduler.step()
            model.eval()
            with torch.no_grad():
                hit10, hit3, hit1, mrr = self.evaluate(model, d.test_data, save_path=evaluation_ranks_file)
                if mrr > max_mrr:
                    max_mrr = mrr
                    max_it = it
                    bad_cnt = 0
                    torch.save({'model_state_dict': model.state_dict(),}, checkpoint_file)
                    logging.info(f"New best model saved at epoch {it} with MRR: {mrr}")
                else:
                    bad_cnt += 1
                    if bad_cnt == self.early_stop:
                        logging.info(f"Early stopping at epoch {it}")
                        break
            print('Valid Result at it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' %(it, hit10, hit3, hit1, mrr))
            metrics = {'MRR':mrr, 'HIT@1':hit1, 'HIT@3':hit3, 'HIT@10':hit10}
            log_metrics('valid', it, metrics)
            
        with torch.no_grad():
            print("\n--- Final Evaluation on Best Model (Test Set) ---")
            try:
                checkpoint = torch.load(checkpoint_file, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                output_filename = evaluation_ranks_file
                print(f"Saving individual triplet ranks to {output_filename}...")
                hit10, hit3, hit1, mrr = self.evaluate(
                    model, 
                    d.test_data,
                    save_path=output_filename
                )
            except FileNotFoundError:
                print("Could not find 'best_checkpoint.pth'. Evaluating with the last model.")
                hit10, hit3, hit1, mrr = self.evaluate(model, d.test_data, save_path="evaluation_ranks.txt")
        
        print(
            'Final Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' %
            (max_it, hit10, hit3, hit1, mrr))
        
        metrics = {'MRR':mrr, 'HIT@1':hit1, 'HIT@3':hit3, 'HIT@10':hit10}
        log_metrics('final_test', max_it, metrics)

        return mrr, hit1, hit3, hit10