import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import os
from tqdm import tqdm
from torch_geometric.datasets import Twitch, AttributedGraphDataset, CitationFull, WikipediaNetwork, Planetoid, HGBDataset
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import dropout_adj, to_networkx
from models.Model_Hetero import *
from models.MLP_Hetero import MLPSamples
from utils import parameter_parser
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from torch_geometric.transforms import RandomNodeSplit

from copy import deepcopy, copy
from torch_geometric.utils import subgraph

#torch.manual_seed(13)
args = parameter_parser()

    
def save_file(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
def open_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def e_loss(data, x):
    r"""Computes the loss given positive and negative random walks."""
    loss = 0.0
    for edge_type in data.meta_data[1]:
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

    
def evaluate(x, data, edge_types):
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = 0; ap = 0
    for edge_type in edge_types:
         
        edge_index = data[edge_type]['edge_label_index']
        labels = data[edge_type]['edge_label'].long().cpu().numpy()
        
        s, t = edge_index
        src_type, trg_type = edge_type[0], edge_type[2]
    
        s_emb = x[src_type][s].detach()
        t_emb = x[trg_type][t].detach()
        scores = s_emb.mul(t_emb).sum(dim=-1).cpu().numpy()
        auc += roc_auc_score(y_true=labels, y_score=scores)
        ap += average_precision_score(y_true=labels, y_score=scores)
        
    return auc/len(edge_types), ap/len(edge_types)

    
def get_scores(model, data, data0, target_edge_index, data2= None):
    x = model(deepcopy(data.x_dict))   
    x0 = model(deepcopy(data0.x_dict)) if data0 is not None else x
    
    scores = {}
    for edge_type in target_edge_index.keys():
        edge_index = target_edge_index[edge_type]
        s, t = edge_index
        src_type, trg_type = edge_type[0], edge_type[2]
        #print(src_type, trg_type, edge_type)
        s_emb = x[src_type][s]
        t_emb = x0[trg_type][t]
        scores[edge_type] = s_emb.mul(t_emb).sum(dim=-1)
    return scores, x
    


def update_edge_types_names(data):
    data2 = HeteroData()
    metadata_ = []
    node_types = data.metadata()[0]
    for nt in data.metadata()[0]:
        data2[nt]= data[nt]
        
    
    edge_types = []
    for et in data.metadata()[1]:
        if et[1] != 'to':
            data2[(et[0], 'to', et[2])] = data[et] 
            edge_types.append((et[0], 'to', et[2]))
        else:
            data2[et] = data[et]
            edge_types.append(et)
    
    data2.meta_data =  [node_types, edge_types]
    data2.x_dict= data.x_dict
    return data2


def add_metadata(data):
    data2 = HeteroData()
    metadata_ = []
    node_types = data.metadata()[0]
    
    for nt in data.metadata()[0]:
        data2[nt]= data[nt]
        
    
    edge_types = []
    for et in data.metadata()[1]:
        data2[et] = data[et]
        edge_types.append(et)

    data2.meta_data = [node_types, edge_types]
    data2.x_dict= data.x_dict
    return data2

    
print(args.name)


##### preparing and loading the data

if args.transductive:
     if os.path.isfile('datasplits/'+args.name+'_train_data.pickle'):
        train_data = open_file('datasplits/'+args.name+'_train_data.pickle')
        valid_data = open_file('datasplits/'+args.name+'_valid_data.pickle')
        test_data = open_file('datasplits/'+args.name+'_test_data.pickle')
   
    else:
        if name == "DBLP":
        dataset = HGBDataset(root+name, name)
        data = dataset.data
        data['venue'].x = torch.arange(data['venue'].num_nodes).reshape(-1, 1)
        targets = {"paper"}

        elif name == "ACM":
            dataset = HGBDataset(root+name, name)
            data = dataset.data
            print(data)
            data['term'].x = torch.arange(data['term'].num_nodes).reshape(-1, 1)
            targets = {"paper"}

        elif name == "IMDB":
            dataset = HGBDataset(root+name, name)
            data = dataset.data
            data['keyword'].x = torch.arange(data['keyword'].num_nodes).reshape(-1, 1)
            targets = {"movie"}   


        rlp = RandomLinkSplit(
            edge_types=data.edge_types,
            neg_sampling_ratio=1.0)

        train_data, valid_data, test_data = rlp(data)


        save_file(train_data, path+'datasplits/'+name+'_train_data.pickle')
        save_file(valid_data, path+'datasplits/'+name+'_valid_data.pickle')
        save_file(test_data, path+'datasplits/'+name+'_test_data.pickle')

else:
    if os.path.isfile('datasplits/'+'ind'+args.name+'_train_data.pickle'):
        train_data = open_file('datasplits/'+'ind'+args.name+'_train_data.pickle')
        valid_data = open_file('datasplits/'+'ind'+args.name+'_valid_data.pickle')
        test_data = open_file('datasplits/'+'ind'+args.name+'_test_data.pickle')
        
    else:    
        if name == "DBLP":
            dataset = HGBDataset(root+name, name)
            data = dataset.data
            print(data)
            data["paper"]["y2"] = torch.rand(data["paper"].num_nodes) > 0.5   
            data['venue'].x = torch.rand(data['venue'].num_nodes, 1)#torch.arange(data['venue'].num_nodes).reshape(-1, 1)
            rns = RandomNodeSplit(num_val=0.1, num_test=0.0, key = "y2")
            targets = {"paper"}

        elif name == "ACM":
            dataset = HGBDataset(root+name, name)
            data = dataset.data
            print(data)
            data["paper"]["y"] = torch.rand(data["paper"].num_nodes) > 0.5
            data['term'].x = torch.rand(data['term'].num_nodes, 1) #torch.arange(data['term'].num_nodes).reshape(-1, 1)
            rns = RandomNodeSplit(num_val=0.1, num_test=0.0, key = "y")
            targets = {"paper"}

        elif name == "IMDB":
            #dataset = IMDB(root+name)
            dataset = HGBDataset(root+name, name)
            data = dataset.data
            rns = RandomNodeSplit(num_val=0.1, num_test=0.1)
            data['keyword'].x = torch.rand(data['keyword'].num_nodes, 1)
            targets = {"movie"}   

        splits = rns(data.clone())

        train_nodes = {}
        valid_nodes = {}
        for nt in data.node_types:
            if nt in targets:
                train_nodes[nt] = set(splits[nt].train_mask.nonzero().flatten().numpy())
                valid_nodes[nt] = set(splits[nt].val_mask.nonzero().flatten().numpy())

        data = splits


        ##### create transductive and inductive edges, and fill out the data objects 
        transductive_edges = {et: [] for et in data.edge_types}
        inductive_edges = {et: [] for et in data.edge_types}
        transductive_nodes = {nt: set() for nt in data.node_types}
        inductive_nodes = {nt: set() for nt in data.node_types}

        trans_hdata = HeteroData()        
        ind_hdata = HeteroData()        


        for et in data.edge_types:
            st, _, tt = et
            for u, v in data[et].edge_index.numpy().T:

                if st in valid_nodes and u in valid_nodes[st]:
                    if tt not in targets or (tt in targets and v not in valid_nodes[tt]):
                        inductive_edges[et].append((u, v))
                        inductive_nodes[st].add(u)
                        transductive_nodes[tt].add(v)


                elif tt in valid_nodes and v in valid_nodes[tt]:
                    if st not in targets or (st in targets and u not in valid_nodes[st]) :
                        inductive_edges[et].append((u, v))
                        transductive_nodes[st].add(v)
                        inductive_nodes[tt].add(u)

                else:
                    transductive_edges[et].append((u, v))
                    if u not in transductive_nodes[st]:
                        transductive_nodes[st].add(u)
                    if v not in transductive_nodes[tt]:
                        transductive_nodes[tt].add(v)

        for k in transductive_nodes:
            transductive_nodes[k] = torch.tensor(list(transductive_nodes[k])).long()
            inductive_nodes[k] = torch.tensor(list(inductive_nodes[k])).long()
            trans_hdata[k].x = data[k].x
            ind_hdata[k].x = data[k].x
            trans_hdata[k].transductive_nodes = transductive_nodes[k]
            ind_hdata[k].inductive_nodes = inductive_nodes[k]

        for k in transductive_edges:
            transductive_edges[k] = torch.tensor(list(transductive_edges[k])).long().T
            inductive_edges[k] = torch.tensor(list(inductive_edges[k])).long().T
            trans_hdata[k].edge_index = transductive_edges[k]
            if len(inductive_edges[k])!=0:
                ind_hdata[k].edge_index = inductive_edges[k] 


        ##### create valid and testing from inductive data object         

        rns = RandomNodeSplit(num_val=0.1, num_test=0.0, key='y')    
        ind_hdata[targets[0]]["y"] = torch.rand(ind_hdata[targets[0]].num_nodes) > 0.5

        splits = rns(ind_hdata)

        valid_nodes = splits[targets[0]].train_mask.nonzero().flatten()
        test_nodes = splits[targets[0]].val_mask.nonzero().flatten()

        valid_data = HeteroData()        
        test_data = HeteroData()

        valid_edges = {}
        test_edges = {}

        for et in splits.edge_types:
            valid_edges[et] = []
            test_edges[et] = []

            for n1, n2 in splits[et].edge_index.T:
                  if et[0] == targets[0]:
                        if n1 in valid_nodes:
                            valid_edges[et].append([n1, n2])
                        else:          
                            test_edges[et].append([n1, n2])

                  if et[2] == targets[0]:
                        if n2 in valid_nodes:
                            valid_edges[et].append([n1, n2])
                        else:          
                            test_edges[et].append([n1, n2])

            valid_data[et].edge_index = torch.tensor(valid_edges[et]).T
            test_data[et].edge_index = torch.tensor(test_edges[et]).T

        for nt in splits.node_types:
            valid_data[nt].x = splits[nt].x
            test_data[nt].x = splits[nt].x


        rlp2 = RandomLinkSplit(
             edge_types=trans_hdata.edge_types,
             num_val=0.0, num_test=0.0)

        train_data, _ , _ = rlp2(trans_hdata)
        valid_data, _ , _ = rlp2(valid_data)
        test_data, _ , _ = rlp2(test_data)


        #### save the data
        save_file(train_data, path+'ind'+name+'_train_data.pickle')
        save_file(valid_data, path+ 'ind'+name+'_valid_data.pickle')
        save_file(test_data, path+'ind'+name+'_test_data.pickle')

    
transductive_dict = None
if args.name == "DBLP":
    transductive_dict = {'venue': train_data['venue'].num_nodes}
    target_node_type = ["paper"]    
elif args.name == "IMDB":
    transductive_dict = {'keyword': train_data['keyword'].num_nodes}
    target_node_type = ["movie"]
elif args.name == "ACM":
    transductive_dict = {'term': train_data['term'].num_nodes}
    target_node_type = ["paper"]
        


if args.name == "IMDB":        
    train_data = update_edge_types_names(train_data)
    valid_data = update_edge_types_names(valid_data)
    test_data  = update_edge_types_names(test_data)
else:
    train_data = add_metadata(train_data)
    valid_data = add_metadata(valid_data)
    test_data  = add_metadata(test_data)

    

print(train_data)    
######### sampling anchors and retrieve the new nodes
num_targets = 4
num_inputs = 4

#if num_targets > 0 and num_inputs > 0:
    
mlp_samples = MLPSamples(train_data, metadata = train_data.meta_data, target_node_type = target_node_type, num_inputs=num_inputs, num_targets=num_targets)
mlp_samples.sample()        

msg_edge_index = {}
for et in tqdm(train_data.meta_data[1]):
    edges = []
    for u, v in train_data[et].edge_index.T:
        if (et[0] in target_node_type and u in set(mlp_samples.inputs[et[0]])) or (et[2] in target_node_type and u in set(mlp_samples.inputs[et[2]])):
            continue
        if (et[0] in target_node_type and v in set(mlp_samples.inputs[et[0]])) or (et[2] in target_node_type and v in set(mlp_samples.inputs[et[2]])):
            continue
            
        edges.append(torch.tensor([[u, v]]))
    msg_edge_index[et] = torch.cat(edges).T 
   

if args.transductive:
    mlp_samples_eval = MLPSamples(valid_data, metadata = valid_data.meta_data, target_node_type = target_node_type,  num_inputs=num_inputs, num_targets=num_targets)
    mlp_samples_eval.sample(inputs = mlp_samples.inputs, targets = mlp_samples.targets)                

    mlp_samples_test = MLPSamples(test_data, metadata = test_data.meta_data, target_node_type = target_node_type,  num_inputs=num_inputs, num_targets=num_targets)
    mlp_samples_test.sample(inputs = mlp_samples.inputs, targets = mlp_samples.targets)                       

else:
    num_valid_nodes_input = valid_data[target_node_type[0]].x.shape[0]# + valid_data[target_node_type[1]].x.shape[0]  #+ valid_data[target_node_type[2]].x.shape[0] 
    num_test_nodes_input = test_data[target_node_type[0]].x.shape[0] #+ test_data[target_node_type[1]].x.shape[0] 

    mlp_samples_eval = MLPSamples(valid_data, metadata = valid_data.meta_data, target_node_type = target_node_type, num_targets=num_targets)
    mlp_samples_eval.sample(targets = mlp_samples.targets)                

    mlp_samples_test = MLPSamples(test_data, metadata = test_data.meta_data, target_node_type = target_node_type, num_targets=num_targets)
    mlp_samples_test.sample(targets = mlp_samples.targets)               

    
anchors = mlp_samples.targets

train_data = train_data.to(args.device)
valid_data = valid_data.to(args.device)
test_data = test_data.to(args.device)

train_data.meta_data[1] = [tuple(x) for x in train_data.meta_data[1]]
valid_data.meta_data[1] = [tuple(x) for x in valid_data.meta_data[1]]
test_data.meta_data[1] = [tuple(x) for x in test_data.meta_data[1]]


###################################, 
connector_model = connector_model(train_data.num_features, 128, train_data.meta_data[0], transductive_dict).to(args.device)
optim = torch.optim.Adam(connector_model.parameters(), lr=0.1)
best = 0.0
for epoch in range(400):    

    connector_model.train()
    x = connector_model(deepcopy(train_data.x_dict))   
          
    loss =  e_loss(train_data, x)  
    
            
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    # validation
    with torch.no_grad():
        connector_model.eval()
        
        auc, ap = evaluate(x, valid_data, valid_data.meta_data[1])
        print(f"Epoch {epoch + 1}/{args.epochs}, Training loss: {loss.item():.10}, Validation AUC: {auc}, AP: {ap}")
    
        if ap > best :
            best = ap
            best_connector = deepcopy(connector_model)
       
print()
print()
    

####################################

## training leap    
mse_loss = nn.MSELoss()
cce = nn.CrossEntropyLoss()
#connector_model = connector_model(train_data.num_features, 128, train_data.meta_data[0], transductive_dict).to(args.device)
model = Model(metadata=train_data.meta_data, transductive_types = transductive_dict).to(args.device)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
best_connector.eval()
   
best = 0.0
best_model = None
 
for epoch in range(args.epochs):    

    model.train()
    #connector_model.train()
    
    #if num_targets > 0 and num_inputs > 0:
        
    target_edge_index = mlp_samples.target_edges
    target_edge_weights, x_mlp = get_scores(best_connector, train_data, None, target_edge_index)

    x, pred = model(x = train_data.x_dict, 
                message_edge_index = msg_edge_index, 
                target_edge_index = target_edge_index,
                target_edge_weights = target_edge_weights)

#     else:
#         target_edge_weights = torch.tensor([])
#         x, pred = model(x = train_data.x_dict, message_edge_index = train_data.edge_index.to(args.device))
        
        
    loss =  model.loss(train_data, x)  
    
    # if num_targets > 0 and num_inputs > 0:
    #     loss +=  0.005 * cce(target_edge_weights.flatten(), target_weight.flatten())

            
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    # validation
    with torch.no_grad():
        model.eval()
        connector_model.eval()
            
        if num_targets > 0 and num_inputs > 0:

            target_edge_index = mlp_samples_eval.target_edges
            target_edge_weights, x_mlp = get_scores(best_connector, train_data, valid_data, target_edge_index)

            out, pred = model(x=valid_data.x_dict,
                            message_edge_index=msg_edge_index,#.edge_index,
                            target_edge_index=target_edge_index,
                            target_edge_weights = target_edge_weights)


        else:
            out, pred = model(x=valid_data.x_dict, message_edge_index=train_data.edge_index.to(args.device))

        
        auc, ap = evaluate(out, valid_data, valid_data.meta_data[1])
        print(f"Epoch {epoch + 1}/{args.epochs}, Training loss: {loss.item():.10}, Validation AUC: {auc}, AP: {ap}")
    
        if ap > best :
            best = ap
            best_model = deepcopy(model)
            #best_connector = deepcopy(connector_model)
           
    #save_file(best_model, 'ind_model'+args.name+'.pickle')

#####################   
# testing 

with torch.no_grad():
        best_model.eval()
        best_connector.eval()
        
    
        if num_targets > 0 and num_inputs > 0:     
                
            target_edge_index = mlp_samples_test.target_edges
            target_edge_weights, _ = get_scores(best_connector, train_data, test_data, target_edge_index)

            out, pred = best_model(x=test_data.x_dict,
                            message_edge_index=msg_edge_index,
                            target_edge_index=target_edge_index,
                            target_edge_weights = target_edge_weights)

        else:
            out, pred = best_model(x=test_data.x, message_edge_index=train_data.edge_index.to(args.device))
            
        auc, ap = evaluate(out, test_data, test_data.meta_data[1])
        print(f"Epoch {epoch + 1}/{args.epochs}, Training loss: {loss.item():.10}, testing AUC: {auc}, AP: {ap}")
    
                      