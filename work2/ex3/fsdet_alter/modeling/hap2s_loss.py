#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
# import import_ipynb
from fsdet_alter.modeling.ohem import hard_example_mining, hard_aware_point_2_set_mining
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


def euclidean_distance(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    #sqrt((x-y)^2)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    return dist.clamp(min=1e-12).sqrt()  # for numerical stability


# In[ ]:


class SpreadLoss(nn.Module):
    def __init__(self, num_classes, m=0.5):
        super(SpreadLoss, self).__init__()
        self.num_classes = num_classes
        self.m = m

    def forward(self, x, target, **pars):  # x:b,10 target:b
        m = self.m
        if len(pars) > 0:
            m = pars['extra_pars'][0]

        one_shot_target = torch.eye(self.num_classes).index_select(dim=0, index=target.data.cpu()).to(x.device)
        a_t = torch.sum(x * one_shot_target, dim=1)
        zeros = torch.zeros(x.size()).to(x.device)
        loss = torch.sum((torch.max(zeros, m - (a_t[:, None] - x))) ** 2, dim=1) - (m**2)
        return torch.mean(loss)

        # a_t = torch.Tensor([x[i][target[i]] for i in range(b)])  # b
        # a_t_stack = a_t.view(b, 1).expand(b, self.num_classes).contiguous().to(x.device) # b, num_classes
        # u = m - (a_t_stack - x)  # b,10
        # mask = u.ge(0).float()  # max(u,0) #b,10
        # loss = ((mask * u) ** 2).sum() / b # NM => ???? - m ** 2  # float
        # return loss


# In[ ]:


class TripletLoss(nn.Module):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    def __init__(self, margin=None, process_dists=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.process_dists = process_dists
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, x, y, z=None):
        """
        Args:
        dist_ap: pytorch Variable, distance between anchor and positive sample,
        shape [N]
        dist_an: pytorch Variable, distance between anchor and negative sample,
        shape [N]
        Returns:
        loss: pytorch Variable, with shape [1]
        """
        if not self.process_dists:
            d_ap = F.pairwise_distance(x, y, 2)
            d_an = F.pairwise_distance(x, z, 2)
        else:
            d_ap = x
            d_an = y
        Y = d_an.data.new().resize_as_(d_an.data).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(d_an, d_ap, Y)
        else:
            loss = self.ranking_loss(d_an - d_ap, Y)
        return loss, d_ap, d_an


# In[ ]:


def isnan(x):
    return x != x

#TripletLoss layer 
class TripletLossLayer(torch.nn.Module):
    def __init__(self,alpha):
        super(TripletLossLayer, self).__init__()
        self.ALPHA = alpha
        self.ranking_loss = nn.SoftMarginLoss()
        
    def triplet_loss(self,a,p,n):
        
        p_dist =  F.pairwise_distance(a, p, 2)
        n_dist = F.pairwise_distance(a,n,2)
        Y = p_dist.data.new().resize_as_(p_dist.data).fill_(1)
        loss = self.ranking_loss(p_dist-n_dist+self.ALPHA,Y)
        return [loss,p_dist,n_dist]
    
    def forward(self,a,p,n):
        loss, p_dist, n_dist = self.triplet_loss(a,p,n)
        self.loss = loss
        return loss, p_dist, n_dist


# In[ ]:


class HardMiningTripletLoss(nn.Module):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        super(HardMiningTripletLoss, self).__init__()
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, feats, targets):
        """
        Args:
        feats: pytorch Variable, features for the targets, shape [NxD]
        targets: pytorch Variable, target values, shape [Nx1]
        Returns:
        loss: pytorch Variable, with shape [1]
        """
        # All pairwise distances
        D = euclidean_distance(feats, feats)

        # Compute hard distances..
        d_ap, d_an = hard_example_mining(D, targets)

        # Compute loss
        Y = d_an.data.new().resize_as_(d_an.data).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(d_an, d_ap, Y)
        else:
            loss = self.ranking_loss(d_an - d_ap, Y)
        return loss


# In[ ]:


class HAP2STripletLoss(nn.Module):
#"paper loss"
    def __init__(self, margin=1, coeff=10, weighting='poly'):
        super(HAP2STripletLoss, self).__init__()
        self.coeff = coeff
        self.weighting = weighting
        self.margin = margin
        if margin is None:
            self.ranking_loss = nn.SoftMarginLoss()
        else:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, targets):
    #"feats embedding immagine"        
        # All pairwise distances
        D = euclidean_distance(feats,feats)

        # Compute hard aware point to set distances..
        d_ap, d_an = hard_aware_point_2_set_mining(D, targets, self.weighting, self.coeff)
        d_ap.requires_grad_()
        d_an.requires_grad_()

        # Compute loss
        Y = (d_an.data.new().resize_as_(d_an.data).fill_(1))
        Variable(Y,requires_grad=True)
        if self.margin is None:
            loss = self.ranking_loss(d_an-d_ap, Y)
        else:
            loss = self.ranking_loss(d_an, d_ap, Y)
        return loss


# In[ ]:


class ClusterLoss(nn.Module):
    def __init__(self,alpha=0.2):
        super(ClusterLoss,self).__init__()
        self.alpha = alpha
        self.ranking_loss = nn.SoftMarginLoss()
        
        self.clusters_sum = []
        self.clusters_count = []
        self.clusters_labels = []
        
    def forward(self,feats,targets):       
       
        t_intra,D_intra = self.Euclidean_intra(feats,targets)
        t_inter,D_inter = self.Euclidean_inter(targets)
        
        Y = (torch.Tensor(t_intra).data.new().resize_as_(torch.Tensor(t_intra).data).fill_(1))
        Y = Y.to(device)
        loss = self.ranking_loss((D_intra-D_inter)+self.alpha,Y)
        
        return loss
        
    def mean_feats(self,feats,targets):
        N = feats.size(0)
        
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        
        target_batch = []
        for i in range(N):
            t = targets[i]
            if not t in target_batch:
                a = feats[is_pos[:,i],:]#list of features computed over the same individual
                sum_a = torch.sum(a,dim=0)              

                if self.clusters_sum:
                    if self.clusters_labels:                       
                        if t in self.clusters_labels:
                            j = self.clusters_labels.index(t.item())
                            self.clusters_sum[j] += sum_a
                            self.clusters_count[j] += a.size(0)
                        else:
                            self.clusters_sum.append(sum_a)
                            self.clusters_labels.append(t)
                            self.clusters_count.append(a.size(0))
                    else: 
                        print('There are no labels {}'.format(M_label))
                else:

                    self.clusters_sum.append(sum_a)
                    self.clusters_labels.append(t.item())
                    self.clusters_count.append(a.size(0))
                target_batch.append(t)
                
    def mean_feats_compute(self):
       
        M_emb = [sum_a/len_a for sum_a,len_a in zip(self.clusters_sum,self.clusters_count)]
        self.M_emb = torch.stack(M_emb).to(device)
                
    def Euclidean_intra(self,feats,targets):
        
        M_intra = self.M_emb
        D = losses.euclidean_distance(feats,M_intra)

        N = feats.size(0)
                       
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        
        target_intra = []
        D_intra = []
        for i in range(N):          
            if not targets[i].item() in target_intra:
                D_id = D[is_pos[:,i],:]            
                target_intra.append(targets[i].item())
                index_mean = self.clusters_labels.index(targets[i].item())
                D_intra.append(torch.max(D_id[:,index_mean]))
        
        D_intra = torch.stack(D_intra)
        D_intra = D_intra.to(device)
        return target_intra, D_intra
    
    def Euclidean_inter(self,targets):
        M_intra = self.M_emb
        N = targets.size(0)
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        target_inter = []
        D_inter= []
        for i in range(N):
            if not targets[i].item() in target_inter:
                index_mean = self.clusters_labels.index(targets[i].item())
                M = M_intra[index_mean,:]  
                target_inter.append(targets[i].item())  
                list_inter = []
                for j in range(len(M_intra)):
                    if not j == index_mean:
                        X = (M_intra[j,:])
                        list_inter.append(torch.pairwise_distance(M.unsqueeze(1),X.unsqueeze(1),2))
                D_inter.append(torch.min(torch.stack(list_inter)))

        
        D_inter = torch.Tensor(D_inter)
        D_inter = D_inter.to(device)
        D_inter.requires_grad_()
        return target_inter, D_inter
    
    def classification(self, feats):
        M_intra = self.M_emb
        D = losses.euclidean_distance(feats,M_intra)
        
        N = feats.size(0)
        classification = []
        for j in range(N):
            classification.append(D[j,:].index(torch.min(D[j,:])))
            
        return classification

