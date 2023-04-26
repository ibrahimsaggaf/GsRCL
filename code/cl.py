import time
import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DL

import config
from networks import Encoder
from validators import EncoderValidator
from utils import merge_dict
from losses import SIMCLRLoss, SUPCONLoss


class ContrastiveLearning:
    def __init__(self, data, metric, folds, id_, dim_enc, in_dim_proj, dim_proj, out_dim, 
                 epochs, step, c_loss, batch_size, aug_method, aug_func, dropout):
        self.data = data
        self.metric = metric
        self.folds = folds
        self.id_ = id_
        self.dim_enc = dim_enc
        self.in_dim_proj = in_dim_proj
        self.dim_proj = dim_proj
        self.out_dim = out_dim
        self.epochs = epochs[1] if c_loss == 'supcon' else epochs[0]
        self.step = step
        self.c_loss = c_loss
        self.batch_size = batch_size
        self.aug_method = aug_method
        self.aug_func = aug_func
        self.dropout = dropout

        self.results_ = {}


    def _weights_init(self, net):
        if type(net) == torch.nn.Linear:
            net.weight.data.normal_(0.0, 0.02)
            net.bias.data.fill_(0)


    def _get_loss_func(self, rank):
        if self.c_loss == 'simclr':
            c_loss_func = SIMCLRLoss(rank)

        elif self.c_loss == 'supcon':
            c_loss_func = SUPCONLoss(rank)

        else:
            raise NotImplementedError(f'The {self.c_loss} loss is not implemeneted')

        return c_loss_func


    def _fit(self, idx, train_fold, rank, q):
        results = {
            idx: {
                'encoder_validation': {}
            }
        }
        c_loss_list = []
        train_X = torch.tensor(self.data.train_X[train_fold], dtype=torch.float32)
        train_y = torch.tensor(self.data.train_y[train_fold], dtype=torch.int32)
        batch_training = train_X.size(0) > self.batch_size
        train_data = torch.utils.data.TensorDataset(train_X, train_y)
        train_loader = DL(
            train_data, 
            batch_size=self.batch_size if batch_training else train_X.size(0)
        )
        val_X = torch.tensor(self.data.val_X, dtype=torch.float32)

        encoder = Encoder(train_X.size(1), self.dim_enc, self.in_dim_proj, self.dim_proj, self.out_dim, 
                          batchnorm=batch_training, dropout=self.dropout, frozen=False)

        encoder.apply(self._weights_init)
        encoder = encoder.to(rank)
        opt_enc = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=1e-6)
        c_loss_func = self._get_loss_func(rank).to(rank)

        encoder_validator = EncoderValidator(self.data, train_fold, self.metric, self.folds, idx, self.id_, 
                                             self.c_loss, self.aug_method)

        encoder.train()
        for epoch in range(self.epochs + 1):
            for x, y in train_loader:
                encoder.zero_grad()
                
                if self.aug_func is not None:
                    aug_i, y = self.aug_func.augment(idx, x, y)
                    aug_j, _ = self.aug_func.augment(idx, x, y)

                else:
                    aug_i = x.detach().clone()
                    aug_j = x.detach().clone()

                aug_i = aug_i.requires_grad_(True).to(rank)
                aug_j = aug_j.requires_grad_(True).to(rank)
                
                proj_i = encoder(aug_i)
                proj_j = encoder(aug_j)
                
                if self.c_loss == 'supcon':
                    c_loss = c_loss_func(proj_i, proj_j, y)
                
                else:
                    c_loss = c_loss_func(proj_i, proj_j)

                c_loss.backward()
                opt_enc.step()

            c_loss_list.append(c_loss.item())

            if epoch % self.step == 0:
                params = encoder.state_dict()
                frozen_params = params.copy()
                for key in params.keys():
                    if key.startswith('proj_head'):
                        del frozen_params[key]
                
                frozen_kwargs = {
                    'in_dim': train_X.size(1), 'dim_enc': self.dim_enc, 'in_dim_proj': self.in_dim_proj, 
                    'dim_proj': self.dim_proj, 'out_dim': self.out_dim, 'batchnorm': batch_training, 
                    'dropout': self.dropout, 'frozen': True
                }
                frozen_encoder = Encoder(**frozen_kwargs)
                frozen_encoder.load_state_dict(frozen_params)
                frozen_encoder.eval()

                with torch.no_grad():
                    train_h = frozen_encoder(train_X)
                    val_h = frozen_encoder(val_X)

                results[idx]['encoder_validation'].update({
                    epoch: encoder_validator.validate(train_h, val_h, frozen_params, frozen_kwargs, epoch),
                })

        best_encoder = encoder_validator.get_best_encoder(results[idx]['encoder_validation'])
        results[idx].update({
            'closs': c_loss_list,
            'best': best_encoder
        })
        
        torch.cuda.empty_cache()

        q.put(results)


    def mp_fit(self):
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        ranks = list(range(config.WORLD_SIZE))
        chunks = []
        for idx, (train_fold, _) in enumerate(self.folds.split(self.data.train_X, self.data.train_y)):
            rank = ranks.pop(0)
            chunks.append((idx, train_fold, rank, q))
            ranks.append(rank)

        processes, results = [], []
        for i in range(config.WORLD_SIZE):
            p = mp.Process(target=self._fit, args=chunks[i])
            p.start()
            processes.append(p)

        for _ in range(config.WORLD_SIZE):
            results.append(q.get())

        for p in processes:
            p.join()

        while True:
            if not processes[0].is_alive():
                p = mp.Process(target=self._fit, args=chunks[-1])
                p.start()
                results.append(q.get())
                p.join()
                break
            time.sleep(5)

        self.results_.update(merge_dict(results))


class CLAugmentation:
    def __init__(self, aug_method):       
        if aug_method == 'noise':
            self.aug_func = self._add_noise

        else:
            raise NotImplementedError(f'The {aug_method} mehtod is not implemeneted')


    def _add_noise(self, args):
        _, X, y = args
        return  X + torch.normal(mean=0, std=1, size=X.size()), y


    def augment(self, *args):
        return self.aug_func(args)
