
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
        self.linear_layer4=nn.Linear(100,50)
        self.linear_layer5=nn.Linear(50,10)
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
        g.ndata['hidden rep']                     = self.activation1(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.linear_layer3(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.activation1(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.linear_layer3(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.activation1(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.linear_layer3(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.activation1(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.linear_layer3(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.activation1(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = self.linear_layer3(g.ndata['hidden rep'])
        ### batchnorm1d
        g.ndata['hidden rep']                     = self.batchnorm(g.ndata['hidden rep'])
        g.ndata['hidden rep']                     = torch.sum(g.ndata['hidden rep'],dim=1)
        g.ndata['hidden rep']                     = self.linear_layer4(g.ndata['hidden rep'])
        out                                       = self.linear_layer5(g.ndata['hidden rep'])
        
        return out
        
        
