
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d,ReLU,MaxPool2d,Linear
import dgl


#this function is the edge update function - 

class EdgeNetwork(nn.Module):
    def __init__(self,inputsize,hidden_layer_size,output_size):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(inputsize, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_size)
        )

        
    def forward(self, x):
        
        #x.dst['node_features'], x.src['node_features'] x.data['distance'] ....

        # put together the node_features, node_hidden_rep, edge distance, and node prediction
                              
        input_data = torch.cat((x.dst['node_features'],
                            x.dst['node_hidden_rep'],
                            x.src['node_features'],
                            x.dst['node_hidden_rep'],
                            torch.unsqueeze(x.data['distance'],1)), 1)
        
        #use a neural network to create an edge hidden representation - 
        output = self.net(input_data)
        
        #you return a dictionary with what you want to "send" to the reciving node


        return {'edge hidden representation': output }

    
class NodeNetwork(nn.Module):
    def __init__(self,inputsize,hidden_layer_size,output_size):
        super().__init__()

        
        self.net = nn.Sequential(
            nn.Linear(inputsize, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_size)
            )



    def forward(self, x):
        
        # this time your input x has:
        # x.mailbox['edge hidden representation'] -> this is what you send with the edge update function above - 
        # it will have the size of the node neighborhood - 
        # (Batch size, number of nodes in neighborhood, edge hidden rep size), so you need to sum/mean over dim=1 
        # x.data['node_hidden_state'] and x.data['node_features'] (this is the existing state of your node)
        # you need to torch.cat the message sum, node hidden state, and node features 
        #- and then apply some fully connected neural network
        
        # return a new hidden state for the node
        
        x.mailbox['edge hidden representation'] # what shape does this have? 
        print("edge hidden representation  shape", x.mailbox['edge hidden representation'].shape)

        message_sum = torch.sum(x.mailbox['edge hidden representation'],dim=1) # what dimension are you summing over?
        
        # the current hidden rep of the nodes
        x.data['node_hidden_rep']
        print("node hidden representation  shape", x.data['node_hidden_rep'].shape)


        input_data = torch.cat([message_sum, x.data['node_features'], x.data['node_hidden_rep']], dim=1)
        
        
        out = self.net(input_data)

        
        return {'node_hidden_rep': out }




class Classifier(nn.Module):
    def __init__(self,inputsize,hidden_layer_size,output_size):
        super().__init__()

        # a network to init the hidden rep of the nodes
        self.node_init = nn.Sequential(
                       nn.Linear(inputsize,hidden_layer_size),
                       nn.ReLU(),
                       nn.Linear(hidden_layer_size,hidden_layer_size),
                       nn.ReLU(),
                       nn.Linear(hidden_layer_size, output_size))
        
        self.edge_network    = EdgeNetwork(inputsize,hidden_layer_size,output_size)
        self.node_network    = NodeNetwork(inputsize,hidden_layer_size,output_size)
        self.edge_classifier = EdgeNetwork(inputsize,hidden_layer_size,output_size) # remember the output size should be configurable!
        self.node_classifier = nn.Sequential(nn.Linear(inputsize,hidden_layer_size,output_size),
                               nn.ReLU(),
                               nn.Linear(hidden_layer_size,output_size))
        
        
        
        
        
    def forward(self, g):
        
        g.ndata['node_hidden_rep'] = self.node_init(g.ndata['node_features'])

        g.edata['prediction'] = torch.zeros(g.num_edges(),device=g.device)
        g.ndata['prediction'] = torch.zeros(g.num_nodes(),device=g.device)
        
       
        for i in range( 10 ):
            
            # call the g.update_all function with the edge and node network
            
            g.update_all(self.edge_network,self.node_network)

            g.apply_edges(self.edge_classifier)
            
            prediction = g.edata['edge hidden representation'].view(-1)
            g.edata['prediction']+= prediction
                
                
            g.ndata['prediction']+= self.node_classifier( g.ndata['edge hidden representation'] ).view(-1)
                
  


        
        
