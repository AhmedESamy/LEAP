from torch_sparse import SparseTensor

from torch_geometric.datasets import CitationFull, Planetoid, Twitch, Amazon, Coauthor
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import to_networkx, subgraph, degree
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv

import torch.nn.functional as F 
import torch.nn as nn
import torch

import networkx as nx
import numpy as np
from utils import parameter_parser

torch.manual_seed(10)
args = parameter_parser()




class MLPSamples:
    
    def __init__(self, data, metadata, target_node_type, num_inputs=10, num_targets=100):
        self.num_inputs = num_inputs
        self.num_targets = num_targets
        self.target_node_type = target_node_type
        self.metadata = metadata
        self.data = data
        self.nodes = {'inductive': []}
        
        # for et in metadata[1]:   
        #     if et[2] not in self.nodes.keys():
        #         self.nodes[et[2]] = (self.data[et].edge_index[1]).tolist()
        #     else:
        #         self.nodes[et[2]].extend(self.data[et].edge_index[1].tolist())
        
     
                
    def sample(self, inputs= None, targets = None):    
        if targets is not None:
            for et in self.metadata[1]:
                if et[0] in self.target_node_type:
                    self.nodes['inductive'].extend(self.data[et].edge_index[0].tolist())
                else:
                    if et[0] not in self.nodes.keys():
                        self.nodes[et[0]] = (self.data[et].edge_index[0]).tolist()
                    else:
                        self.nodes[et[0]].extend(self.data[et].edge_index[0].tolist())

                if et[2] in self.target_node_type and et[0] not in self.target_node_type:
                    self.nodes['inductive'].extend(self.data[et].edge_index[1].tolist())       
                else:
                    if et[2] not in self.nodes.keys():
                        self.nodes[et[2]] = (self.data[et].edge_index[1]).tolist()
                    else:
                        self.nodes[et[2]].extend(self.data[et].edge_index[1].tolist())

            self.nodes['inductive'] = torch.tensor(list(set(self.nodes['inductive'])))
            
        else:       
            for et in self.metadata[1]:   
                if et[0] not in self.nodes.keys():
                    self.nodes[et[0]] = (self.data[et].edge_index[0]).tolist()
                else:
                    self.nodes[et[0]].extend(self.data[et].edge_index[0].tolist())

                if et[2] not in self.nodes.keys():
                    self.nodes[et[2]] = (self.data[et].edge_index[1]).tolist()
                else:
                    self.nodes[et[2]].extend(self.data[et].edge_index[1].tolist())

        for nt in self.metadata[0]:
            if nt in self.nodes.keys():
                self.nodes[nt] = torch.tensor(list(set(self.nodes[nt])))
       
        
        
        if inputs is None: # training
            if targets is None:
                
                self.inputs = {}
                self.targets= {}
                inputs_size_per_node_type = int(self.num_inputs / len(self.metadata[0]))
                targets_size_per_node_type = int(self.num_targets/ len(self.metadata[0]))
                
                #samples = torch.randperm(len(self.data[self.target_node_type].x)) 
                
                #self.targets[self.target_node_type] = samples[:self.num_targets]
                #self.inputs[self.target_node_type] = samples [self.num_targets: self.num_targets + inputs_size_per_node_type]
                
                
                                
                for nt in self.metadata[0]:
                    samples_indx = torch.randperm(len(self.nodes[nt]))
            
                    if nt in self.target_node_type:
                        samples = self.nodes[nt][samples_indx][:inputs_size_per_node_type+ targets_size_per_node_type]
                        self.targets[nt]= samples[:targets_size_per_node_type]
                        self.inputs[nt] = samples [-inputs_size_per_node_type:]
                        
                    else:
                        samples = self.nodes[nt][samples_indx][:targets_size_per_node_type]
                        self.targets[nt]= samples[:targets_size_per_node_type]
                        #self.inputs[nt] = samples [-inputs_size_per_node_type:]
            
            else:
                self.inputs = {}
                self.inputs[self.target_node_type[0]] = self.nodes['inductive']
                #inputs_size_per_node_type = int(self.num_inputs / len(self.metadata[0]))
                
                #samples = torch.randperm(len(self.data[self.target_node_type].x)) 
                
                #self.targets[self.target_node_type] = samples[:self.num_targets]
                #self.inputs[self.target_node_type] = samples [self.num_targets: self.num_targets + inputs_size_per_node_type]
                                
                # for nt in self.metadata[0]:
                #     self.inputs[nt] = self.nodes[nt]
                         
                self.targets = targets
 
        else:
            self.targets= targets
            self.inputs = inputs
            
        self._create_target_edges()
   

    def _create_target_edges(self):
        self.target_edges = {}
        # print(self.targets)
        # print(self.inputs)
        
        # print(self.targets)
        # print(self.inputs)
        for et in self.metadata[1]:
            #print(et)
            #if et[0] in self.target_node_type: #== self.target_node_type:
                #nt = et[0]
            target_edges_per_edge_type = []
#             if et[0] in self.target_node_type:

#                 for node in self.inputs[et[0]]:
#                     edges = zip(self.targets[et[2]], [node] * self.targets[et[2]].shape[0], strict=True)
#                     target_edges_per_edge_type.append(torch.tensor(list(edges)))

            if et[2] in self.target_node_type:
                #print(self.inputs[et[2]])
                for node in self.inputs[et[2]]:
                    edges = zip(self.targets[et[0]], [node] * self.targets[et[0]].shape[0], strict=True)
                    target_edges_per_edge_type.append(torch.tensor(list(edges)))

            
            
            #print(target_edges_per_edge_type)
            # if torch.cat(target_edges_per_edge_type).shape[0] == 0:
            #     continue
                self.target_edges[et] = torch.stack((torch.cat(target_edges_per_edge_type).T[0], torch.cat(target_edges_per_edge_type).T[1]), dim = 0)
            #print(self.target_edges)
        #print(self.target_edges)
        
            