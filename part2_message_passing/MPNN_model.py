
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
            ... )

        
    def forward(self, x):
        
        
        #x.dst['node_features'], x.src['node_features'] x.data['distance'] ....

        # put together the node_features, node_hidden_rep, edge distance, and node prediction
                              
        input_data = torch.cat([ ...   ],dim=1)
        
        #use a neural network to create an edge hidden represetation - 
        
        #you return a dictionary with what you want to "send" to the reciving node


        return {'edge hidden represetation': output }

    
class NodeNetwork(nn.Module):
    def __init__(self,inputsize,hidden_layer_size,output_size):
        super().__init__()

        
        self.net = nn.Sequential(
            ... )



    def forward(self, x):
        
        # this time your input x has:
        # x.mailbox['edge hidden represetation'] -> this is what you send with the edge update function above - 
        # it will have the size of the node neighborhood - 
        # (Batch size, number of nodes in neighborhood, edge hidden rep size), so you need to sum/mean over dim=1 
        # x.data['node_hidden_state'] and x.data['node_features'] (this is the existing state of your node)
        # you need to torch.cat the message sum, node hidden state, and node features 
        #- and then apply some fully connected neural network
        
        # return a new hidden state for the node

        x.mailbox['edge hidden represetation'] # what shape does this have? 

        message_sum = torch.sum( ... ,dim=1) # what dimension are you summing over?
        
        # the current hidden rep of the nodes
        x.data['node_hidden_rep']


        input_data = torch.cat([message_sum, ... ],dim=1)
        
        
        out = self.net(out)

        
        return {'node_hidden_rep': out }


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        

        # a network to init the hidden rep of the nodes
        self.node_init = nn.Sequential(
            nn.Linear(2,50),nn.ReLU(),nn.Linear(50,node_representation_size))
        
        self.edge_network = EdgeNetwork( ..
        self.node_network = NodeNetwork( ... 
        self.edge_classifier = EdgeNetwork( ... # remember the output size should be configurable!
        self.node_classifier = nn.Sequential( .. 
        
        
        
    def forward(self, g):
        
        g.ndata['node_hidden_rep'] = self.node_init(g.ndata['node_features'])

        g.edata['prediction'] = torch.zeros(g.num_edges(),device=g.device)
        g.ndata['prediction'] = torch.zeros(g.num_nodes(),device=g.device)
        
       
        for i in range( number of GN block iterations ):
            
            # call the g.update_all function with the edge and node network
            
            g.update_all(self.edge_network,self.node_network)

            g.apply_edges( #put the edge classifier in here )
            
            prediction = g.edata['edge hidden represetation'].view(-1)
            g.edata['prediction']+= prediction
                
                
            g.ndata['prediction']+= self.node_classifier( #the input should be the node hidden rep ).view(-1)
                
  


        
        