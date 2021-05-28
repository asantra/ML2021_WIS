
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear,BatchNorm2d, BatchNorm1d
import dgl

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1=nn.Linear(2,100)  #shape - (n_pictues_inbatch,n.points,8)
        self.linear_layer2=nn.Linear(100+100,100) 
        self.linear_layer3=nn.Linear(100,100)
        self.batchnorm    =BatchNorm1d(100)
        self.linear_layer4=nn.Linear(100,70)
        self.linear_layer5=nn.Linear(70,50)
        self.linear_layer6=nn.Linear(50,10)
        self.activation1  =nn.ReLU()

       
    def forward(self,g):
        
        g.ndata['hidden rep'] = self.linear_layer1(g.ndata['xy'])
        mean_of_node_rep      = dgl.mean_nodes(g,'hidden rep')
        broadcasted_sum       = dgl.broadcast_nodes(g,mean_of_node_rep)
        g.ndata['global rep'] = broadcasted_sum
        
        input_to_layer        = torch.cat([
                                  g.ndata['hidden rep'], 
                                  g.ndata['global rep']],dim=1)

        g.ndata['hidden rep'] = self.linear_layer2(input_to_layer)
        
        #### 5 times
        g.ndata['final rep']                     = self.activation1(g.ndata['hidden rep'])
        g.ndata['final rep']                     = self.linear_layer3(g.ndata['final rep'])
        g.ndata['final rep']                     = self.activation1(g.ndata['final rep'])
        g.ndata['final rep']                     = self.linear_layer3(g.ndata['final rep'])
        g.ndata['final rep']                     = self.activation1(g.ndata['final rep'])
        g.ndata['final rep']                     = self.linear_layer3(g.ndata['final rep'])
        g.ndata['final rep']                     = self.activation1(g.ndata['final rep'])
        g.ndata['final rep']                     = self.linear_layer3(g.ndata['final rep'])
        g.ndata['final rep']                     = self.activation1(g.ndata['final rep'])
        g.ndata['final rep']                     = self.linear_layer3(g.ndata['final rep'])
        g.ndata['final rep']                     = self.activation1(g.ndata['final rep'])
        g.ndata['final rep']                     = self.linear_layer3(g.ndata['final rep'])
        g.ndata['final rep']                     = self.activation1(g.ndata['final rep'])
        g.ndata['final rep']                     = self.linear_layer3(g.ndata['final rep'])
        ### batchnorm1d
        g.ndata['final rep']                     = self.batchnorm(g.ndata['final rep'])
        
        
        #### second time the whole thing
        #mean_of_node_rep       = dgl.mean_nodes(g,'final rep')
        #broadcasted_sum        = dgl.broadcast_nodes(g,mean_of_node_rep)
        #g.ndata['global rep2'] = broadcasted_sum
        
        #input_to_layer         = torch.cat([
                                  #g.ndata['final rep'], 
                                  #g.ndata['global rep2']],dim=1)
        #g.ndata['hidden rep2'] = self.linear_layer2(input_to_layer)
        #g.ndata['final rep2']                     = self.activation1(g.ndata['hidden rep2'])
        #g.ndata['final rep2']                     = self.linear_layer3(g.ndata['final rep2'])
        #g.ndata['final rep2']                     = self.activation1(g.ndata['final rep2'])
        #g.ndata['final rep2']                     = self.linear_layer3(g.ndata['final rep2'])
        #g.ndata['final rep2']                     = self.activation1(g.ndata['final rep2'])
        #g.ndata['final rep2']                     = self.linear_layer3(g.ndata['final rep2'])
        #g.ndata['final rep2']                     = self.activation1(g.ndata['final rep2'])
        #g.ndata['final rep2']                     = self.linear_layer3(g.ndata['final rep2'])
        #g.ndata['final rep2']                     = self.activation1(g.ndata['final rep2'])
        #g.ndata['final rep2']                     = self.linear_layer3(g.ndata['final rep2'])
        #### batchnorm1d
        #g.ndata['final rep2']                     = self.batchnorm(g.ndata['final rep2'])
        
        
        #### sum of all of them
        sum_of_node_rep                          = dgl.sum_nodes(g,'final rep')
        sum_of_node_rep                          = self.linear_layer4(sum_of_node_rep)
        sum_of_node_rep                          = self.activation1(sum_of_node_rep)
        sum_of_node_rep                          = self.linear_layer5(sum_of_node_rep)
        sum_of_node_rep                          = self.activation1(sum_of_node_rep)
        out                                      = self.linear_layer6(sum_of_node_rep)
        
        return out
        
        
