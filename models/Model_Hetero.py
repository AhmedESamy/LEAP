from torch_sparse import SparseTensor

from torch_geometric.datasets import CitationFull, Planetoid, Twitch, Amazon, Coauthor
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import to_networkx, subgraph, degree
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GAE, VGAE, HeteroConv, Linear, BatchNorm

import torch.nn.functional as F 
import torch.nn as nn
import torch

import networkx as nx
import numpy as np
from utils import parameter_parser
from copy import copy, deepcopy

#torch.manual_seed(10)
args = parameter_parser()


class connector_model(nn.Module):
    
    def __init__(self, input_dim, output_dim, node_types, transductive_types = None):
        super().__init__()
        self.em_dict = nn.ModuleDict()
        self.lin1 = nn.ModuleDict()
        self.bn1 = nn.ModuleDict()
        self.lin2 = nn.ModuleDict()
        self.bn2 = nn.ModuleDict()
        self.tt = transductive_types
        
        for nt in node_types:
            if args.transductive:
                 if transductive_types is not None and nt in transductive_types:
                    self.em_dict[nt] = nn.Embedding(transductive_types[nt], 128) 
                    nn.init.xavier_uniform_(self.em_dict[nt].weight)
                 else:
                    self.em_dict[nt] = None
                
            self.lin1[nt] = Linear(-1, output_dim)
            self.bn1[nt] = BatchNorm(output_dim)
            self.lin2[nt] = Linear(output_dim, output_dim)
            self.bn2[nt] = BatchNorm(output_dim)
                
    def forward(self, x_dict):
        out = {}
        for node_type in x_dict:
            if args.transductive:
                if self.em_dict[node_type] is not None:
                    x_dict[node_type] = self.em_dict[node_type](x_dict[node_type].to(args.device)).squeeze()
            
            out[node_type] = (self.lin1[node_type](x_dict[node_type].float().to(args.device)))
            out[node_type] = self.bn1[node_type](out[node_type])
#             out[node_type] = (self.lin2[node_type](out[node_type]))
#             out[node_type] = self.bn2[node_type](out[node_type])
            
        return out    
    
    
    
class Encoder(nn.Module):
    def __init__(self, metadata, hidden_channels, transductive_types = None):
        super().__init__()
        self.conv = HeteroConv({edge_type:(GraphConv((-1, -1),  hidden_channels, "mean")) for edge_type in metadata[1]}, aggr = 'sum')
        self.lin = nn.ModuleDict()
        self.relu = torch.nn.PReLU()
        self.em_dict = nn.ModuleDict()
        self.bn = nn.ModuleDict()
              
        for nt in metadata[0]:
            if args.transductive:
                if transductive_types is not None and nt in transductive_types:
                    self.em_dict[nt] = nn.Embedding(transductive_types[nt], 32) 
                    nn.init.xavier_uniform_(self.em_dict[nt].weight)   
                
                
            self.lin[nt] = Linear(-1, hidden_channels)
            self.bn[nt] = BatchNorm(hidden_channels)
            
        
    def forward(self, x_dict, edge_index, edge_weights = None, infer = False):
        device = next(self.parameters()).device
        
        def to_float(x_dict): 
            return {k: v.float() for k, v in x_dict.items()}
        
        def to_device(x_dict): 
            return {k: v.to(args.device) for k, v in x_dict.items()}
        
        if args.transductive: 
            x_dict = to_device(x_dict)
            for node_type in x_dict: 
                if node_type in self.em_dict:
                    x_dict[node_type] = self.em_dict[node_type](x_dict[node_type].long()).squeeze() 
                
        if edge_weights is None:
            x = self.conv(to_device(to_float(x_dict)), edge_index_dict=to_device(edge_index))
        else:
            x = self.conv(to_device(x_dict), edge_index_dict=to_device(edge_index), edge_weight_dict = to_device(edge_weights))
           
        return x
   

    

class Model(nn.Module):
    def __init__(self, metadata, transductive_types, output_dim=128):
        super().__init__()
        self.metadata = metadata
        self.output_dim = output_dim
        self.encode = Encoder(metadata, output_dim, transductive_types = transductive_types)
        self.encode2 = Encoder(metadata, output_dim, transductive_types = transductive_types)
        
    
    def forward(self, x, message_edge_index, target_edge_index=None, target_edge_weights = None):
        """
        x.shape = [N, D]
        message_edge_index = [2, E']
        target_edge_inde  x = [2, M * num_targets]
        mlp_inputs = [M]
        """
        def to_float(x_dict): 
            return {k: v.float() for k, v in x_dict.items()}
        
           
        edge_index = {}
        edge_weight =  {}
        m_edge_weight = {}
        e = {}
        t = {}
        m = {}
       
        for et in self.metadata[1]:
            if et in target_edge_index.keys():
                #print(et, self.metadata[1])
                # if not self.training:
                #     print(et, target_edge_index[et])
                #     print(target_edge_weights[et])

                e[et] = (target_edge_weights[et])
                edge_index[et] = torch.cat((message_edge_index[et], target_edge_index[et]), dim= 1)
                edge_weight[et] = torch.cat((torch.ones(message_edge_index[et].shape[1]).to(args.device), e[et]))   
                
            else:
                edge_index[et] = message_edge_index[et]
                edge_weight[et] = torch.ones(message_edge_index[et].shape[1]).to(args.device)   
               
                #print(et, self.metadata[1])
                
#                 m[reversed_et] = torch.stack((message_edge_index[et][1], message_edge_index[et][0]), dim = 0)            
#                 edge_index[reversed_et] = m[reversed_et]
#                 edge_weight[reversed_et] = torch.ones(m[reversed_et].shape[1]).to(args.device)
                
        if args.transductive:           
            out = self.encode(to_float(x), target_edge_index, target_edge_weights)  
            out2 = self.encode2(to_float(x), message_edge_index)   
            for k, v in out2.items():
                if k in out.keys():
                    out[k] = (out[k] + out2[k])
                else:
                    out[k] = out2[k]
            
        else:
            out = self.encode(deepcopy(x), edge_index, edge_weight) 
            #out = self.encode(deepcopy(x), message_edge_index) 
            
#             out2 = self.encode(deepcopy(x), message_edge_index)   
#             for k in self.metadata[0]:
#                   out[k] = (torch.tensor(out[k]) + torch.tensor(out2[k]))
        return out, target_edge_weights

    
    def e_loss(self, data, x):
        r"""Computes the loss given positive and negative random walks."""
        loss = 0.0
        for edge_type in self.metadata[1]:
            k = list(x.keys())[0]
            edge_index = data[edge_type]['edge_label_index'].to(args.device)
            labels = data[edge_type]['edge_label'].long().to(args.device)
        
            # Positive loss.
            EPS = 0.0000001
            src, trg = edge_index
            src_type, trg_type = edge_type[0], edge_type[2]
        
            src_x = x[src_type][src][labels.bool()]
            trg_x = x[trg_type][trg][labels.bool()]

            h_start = src_x
            h_rest = trg_x

            out = (h_start * h_rest).sum(dim=-1).view(-1)
            pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

            # Negative loss.
            src_x = x[src_type][src][~labels.bool()]
            trg_x = x[trg_type][trg][~labels.bool()]

            h_start = src_x
            h_rest = trg_x

            out = (h_start * h_rest).sum(dim=-1).view(-1)
            neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

            loss += pos_loss + neg_loss
            
            del edge_index, labels

        return loss.mean()
    
    
    def loss(self, data, z_dict: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
                
        e_loss = self.e_loss(data, z_dict)
        return e_loss
    
    
#     def load(self, path="model.pt"):
#         self.load_state_dict(torch.load(path))

#     def save(self, path="model.pt"):
#         torch.save(self.state_dict(), path)

        
        
 
        
        
