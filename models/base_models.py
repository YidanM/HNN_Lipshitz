"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)
    
    def encode0(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode0(x, adj) 
        return h
    
    def encode(self, x, adj):
        output = self.encoder.encode(x, adj) 
        return output

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, drop_weight_all, x_hyp_all, data, split, multiplier, lip):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
    
        # lipschitz regularization
        if self.manifold.name == 'Hyperboloid':
            lipsum = self.lipschitz_bounds_hyperboloid(drop_weight_all, x_hyp_all)
        if self.manifold.name == 'PoincareBall':
            lipsum = self.lipschitz_bounds_poincareball(drop_weight_all, x_hyp_all)
        loss += multiplier * lipsum[lip]

        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'lip1': lipsum[0], 'lip2': lipsum[1]}
        return metrics

    def lipschitz_bounds_hyperboloid(self, drop_weight_all, x_hyp_all):
        for i in range(len(drop_weight_all)):
            drop_weight = drop_weight_all[i]
            x_hyp = x_hyp_all[i]
            W = drop_weight[:, 1:] 
            W_norm = torch.norm(W, p=2)
            x = torch.transpose(x_hyp, 0, 1) 
            x_t = x[0, :]
            x_s = x[1:, :] 
            x_s_norm = torch.norm(x_s, p=2, dim=0)
            a = torch.acosh(torch.tensor(2**0.5))
            Wx_s = torch.matmul(W, x_s)
            Wx_s_norm = torch.norm(Wx_s, p=2, dim=0)
            lip1 = (torch.sinh(a * Wx_s_norm) * (Wx_s_norm - 1) / Wx_s_norm + torch.cosh(a * Wx_s_norm) * (a / Wx_s_norm - a / (x_t * x_s_norm) + 1 / x_s_norm)) * W_norm
            lip2 = torch.exp(a * Wx_s_norm) / Wx_s_norm * W_norm            
            if i == 0:
                lipproduct1 = lip1
                lipproduct2 = lip2
            else:
                lipproduct1 *= lip1
                lipproduct2 *= lip2
        lipsum1 = torch.sum(lipproduct1)
        lipsum2 = torch.sum(lipproduct2)
        return lipsum1, lipsum2
        
    def lipschitz_bounds_poincareball(self, drop_weight_all, x_hyp_all):
        for i in range(len(drop_weight_all)):
            drop_weight = drop_weight_all[i]
            x_hyp = x_hyp_all[i]
            b= torch.norm(torch.tanh(torch.tensor(1)))
            a = 1 / (b ** 2) + 1 / (b * (1 - b ** 2))
            x = torch.transpose(x_hyp, 0, 1) 
            x_norm = torch.norm(x, dim=0)
            M = drop_weight
            M_transpose = torch.transpose(M, 0, 1)
            M_norm = torch.norm(M, p=2)
            Mx = torch.matmul(M, x)
            Mx_norm = torch.norm(Mx, p=2, dim=0)
            tanh_arg = (Mx_norm / x_norm) * torch.atanh(x_norm)
            tanh_term = torch.tanh(tanh_arg)
            sech_term = 1 / torch.cosh(tanh_arg) ** 2
            part1 = 2 * M_norm / Mx_norm 
            atanh_x = torch.atanh(x_norm)
            part2 = atanh_x * torch.norm(torch.matmul(M_transpose, M)) / (Mx_norm ** 2) + (atanh_x) / (x_norm ** 2) + 1 / (x_norm * (1 - x_norm ** 2))
            lip1 = tanh_term * part1 + Mx_norm * sech_term * part2
            lip2 = 2 * M_norm / Mx_norm + torch.norm(torch.matmul(M_transpose, M)) / (Mx_norm) + a * Mx_norm
            if i==0:
                lipproduct1 = lip1
                lipproduct2 = lip2
            else:
                lipproduct1 *= lip1
                lipproduct2 *= lip2      
        lipsum1 = torch.sum(lipproduct1)
        lipsum2 = torch.sum(lipproduct2)
        return lipsum1, lipsum2

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

