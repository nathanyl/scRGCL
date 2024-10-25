import csv

import torch as t
import torch
from torch import nn
import torch.nn.functional as F
import scipy
from copy import deepcopy
import numpy as np
import pandas as pd
import sys
import math
import pickle as pkl
import math
import torch as t 
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.metrics import precision_score,f1_score
from torch_geometric.utils import to_undirected,remove_self_loops
from torch.nn.init import xavier_normal_,kaiming_normal_
from torch.nn.init import uniform_,kaiming_uniform_,constant
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.data import Batch,Data
from collections import Counter 
from torch.utils import data as tdata
from sklearn.model_selection import StratifiedKFold
from fightingcv_attention.attention.ECAAttention import ECAAttention
from torch_geometric.nn import GATConv
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from sklearn.decomposition import PCA
import random
from torch.nn.utils.rnn import pad_sequence




import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import math

#
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)
#
class SAGEConv(MessagePassing):
    # def __init__(self, in_channels, out_channels, normalize=False, bias=True, activate=False, alphas=[0,1], shared_weight=False, aggr='mean',
    #              **kwargs):
    def __init__(self, in_channels, out_channels, normalize=False, bias=True, activate=False, alphas=[0.65, 0.35], shared_weight=False, aggr='mean',
                 **kwargs):
        #
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        # 
        self.shared_weight = shared_weight
        #
        self.activate = activate
        # 
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if self.shared_weight:
            self.self_weight = self.weight 
        else:
            self.self_weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.alphas = alphas #[self_alpha, pro_alpha]
        # 
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)
        uniform(self.in_channels, self.self_weight)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        #
      
        out = torch.matmul(x, self.self_weight)
        
        out2 = self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)
        return self.alphas[0]*out + self.alphas[1] * out2
    # 
    def message(self, x_j,edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1, 1) * x_j

    def update(self, aggr_out):

        if self.activate:
            aggr_out = F.relu(aggr_out)
            #aggr_out = F.hardswish(aggr_out)
            #aggr_out = F.mish(aggr_out)

            
        if torch.is_tensor(aggr_out):
            aggr_out = torch.matmul(aggr_out, self.weight)
        else:
            aggr_out = (None if aggr_out[0] is None else torch.matmul(aggr_out[0], self.weight),
                 None if aggr_out[1] is None else torch.matmul(aggr_out[1], self.weight))
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



from torch.nn.init import normal,uniform_
def init_weights(m):
    if type(m) == nn.Linear:
        # nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.xavier_uniform_(m.weight)
        #nn.init.kaiming_uniform_(m.weight, a=0.001, mode='fan_in', nonlinearity='leaky_relu')
        m.bias.data.fill_(0.01)

def help_bn(bn1,x):
    #         dim1, dim2,dim3 = x.shape 
    x = x.permute(1,0,2)  # #samples x #nodes x #features
    x = bn1(x)
    x = x.permute(1,0,2) # #nodes x #samples x #features
    return x
# 
def sup_constrive(representations, label, T):
    n = label.shape[0]
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
  
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t())) - torch.eye(n, n).to('cuda')

    
    mask_no_sim = torch.ones_like(mask) - mask
    
    mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)
    
    similarity_matrix = torch.exp(similarity_matrix / T)
    

    similarity_matrix = similarity_matrix * mask_dui_jiao_0.to('cuda')
    sim = mask * similarity_matrix
    no_sim = similarity_matrix - sim
    no_sim_sum = torch.sum(no_sim, dim=1)
 
    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)
   
    #print(mask_no_sim.is_cuda)
    loss = mask_no_sim + loss + torch.eye(n, n).to('cuda')
    
    loss = -torch.log(loss)  # 求-log
    # loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  

      loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

    return loss

class WeightFreezing(nn.Module):
    def __init__(self, input_dim, output_dim, shared_ratio=0.3, multiple=0):
        super(WeightFreezing, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        mask = torch.rand(input_dim, output_dim) < shared_ratio
        self.register_buffer('shared_mask', mask)
        self.register_buffer('independent_mask', ~mask)

        self.multiple = multiple

    def forward(self, x, shared_weight):
        combined_weight = torch.where(self.shared_mask, shared_weight*self.multiple, self.weight.t())
        output = F.linear(x, combined_weight.t(), self.bias)
        return output


class scRGCL(nn.Module):
    def __init__(self, in_channel=1, mid_channel=8, out_channel=2, num_nodes=2207, edge_num=151215,
                **args):
        super(scRGCL, self).__init__()
        self.mid_channel = mid_channel
        self.dropout_ratio = args.get('dropout_ratio', 0.3)
        print('model dropout raito:', self.dropout_ratio)
        n_out_nodes = num_nodes
        self.global_conv1_dim = 4*3
        self.global_conv2_dim = args.get('global_conv2_dim', 4)
        input_dim = num_nodes
        self.c_layer = nn.Linear(input_dim, 63, bias=False)
        #self.sa = ScaledDotProductAttention(d_model=8, d_k=8, d_v=8, h=8)
        
        # self.res_gat = GATConv(8, 8, dropout=0.3)
        self.res_conv1 = t.nn.Conv2d(8, 24, [1,1])
        #self.res_bn1 = t.nn.BatchNorm2d(16)
        self.res_bn1 = t.nn.BatchNorm2d(24)
        self.res_act1 = nn.Tanh()

        #self.res_conv2 = t.nn.Conv2d(16, 32, [1,1])
        self.res_conv2 = t.nn.Conv2d(24, 48, [1, 1])
        self.res_bn2 = t.nn.BatchNorm2d(48)
        self.res_act2 = nn.Tanh()

        # self.res_conv3 = t.nn.Conv2d(32, 16, [1,1])
        self.res_conv3 = t.nn.Conv2d(48, 24, [1, 1])
        #self.res_bn3 = t.nn.BatchNorm2d(16)
        self.res_bn3 = t.nn.BatchNorm2d(24)
        self.res_act3 = nn.Tanh()

        #self.res_conv4 = t.nn.Conv2d(16, 8, [1,1])
        self.res_conv4 = t.nn.Conv2d(24, 8, [1, 1])
        self.res_bn4 = t.nn.BatchNorm2d(8)
        self.res_act4 = nn.Tanh()

        self.res_fc = nn.Linear(8, 8)
        
        self.conv1 = SAGEConv(in_channel, 8, )
        self.conv2 = SAGEConv(8, mid_channel,)
       
        self.bn1 = torch.nn.LayerNorm((num_nodes,8))
        self.bn2 = torch.nn.LayerNorm((num_nodes,mid_channel))
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # 1
        self.global_conv1 = t.nn.Conv2d(mid_channel*1, self.global_conv1_dim,[1,1])
        self.global_bn1 = torch.nn.BatchNorm2d(self.global_conv1_dim)
        self.global_act1 = nn.ReLU()
        # 2
        self.global_conv2 = t.nn.Conv2d(self.global_conv1_dim, self.global_conv2_dim,[1,1])
        self.global_bn2 = torch.nn.BatchNorm2d(self.global_conv2_dim)
        self.global_act2 = nn.ReLU()


            
        last_feature_node = 64
        channel_list = [self.global_conv2_dim*n_out_nodes,256,64]
        if args.get('channel_list', False):
            channel_list = [self.global_conv2_dim*n_out_nodes,128]
            last_feature_node = 128

        self.nn = []
        for idx, num in enumerate(channel_list[:-1]):
            self.nn.append(nn.Linear(channel_list[idx], channel_list[idx+1]))
            self.nn.append(nn.BatchNorm1d(channel_list[idx+1]))
            if self.dropout_ratio > 0:
                self.nn.append(nn.Dropout(0.3))
            self.nn.append(nn.ReLU())
        self.global_fc_nn = nn.Sequential(*self.nn)
        self.fc1 = nn.Linear(last_feature_node, out_channel)
        self.classifier = WeightFreezing(last_feature_node, out_channel, shared_ratio=0.35)

        self.shared_weights = nn.Parameter(torch.Tensor(out_channel, last_feature_node), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(last_feature_node))

        nn.init.kaiming_uniform_(self.shared_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.shared_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fixed_weight = self.shared_weights.t() * self.classifier.shared_mask

        self.edge_num = edge_num
        self.weight_edge_flag = True  
        # print('trainalbe edges :',self.weight_edge_flag)
        if self.weight_edge_flag:
            self.edge_weight = nn.Parameter(t.ones(edge_num).float()*0.01)
            # _=normal(self.edge_weight,mean=0,std=0.01)
            # _ = uniform_(self.edge_weight,a=-2,b=2)
        else:
            self.edge_weight = None
    
        self.reset_parameters()
    
    def reset_parameters(self,):
        # uniform(self.mid_channel, self.global_conv1.weight
        # nn.init.kaiming_uniform_(m.weight, a=0.001, mode='fan_in', nonlinearity='leaky_relu')
        # 1
        self.conv1.apply(init_weights)
        # 2
        self.conv2.apply(init_weights)

        
        nn.init.kaiming_normal_(self.res_conv1.weight, mode='fan_out')
        # uniform(8, self.res_conv1.bias)
        uniform(8, self.res_conv1.bias)
        nn.init.kaiming_normal_(self.res_conv2.weight, mode='fan_out')
        # uniform(16, self.res_conv2.bias)
        uniform(24, self.res_conv2.bias)
        nn.init.kaiming_normal_(self.res_conv3.weight, mode='fan_out')
        #uniform(32, self.res_conv3.bias)
        uniform(48, self.res_conv3.bias)
        nn.init.kaiming_normal_(self.res_conv4.weight, mode='fan_out')
        #uniform(16, self.res_conv4.bias)
        uniform(24, self.res_conv4.bias)

        nn.init.kaiming_normal_(self.global_conv1.weight, mode='fan_out')
        #nn.init.kaiming_uniform_(self.global_conv1.weight, a=0.001, mode='fan_out', nonlinearity='leaky_relu')
        uniform(self.mid_channel, self.global_conv1.bias)

        #nn.init.kaiming_uniform_(self.global_conv2.weight, a=0.001, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.global_conv2.weight, mode='fan_out')
        uniform(self.global_conv1_dim, self.global_conv2.bias)

        self.global_fc_nn.apply(init_weights)
        self.fc1.apply(init_weights)
        # uniform(self.in_channels, self.bias)
        # uniform(self.in_channels,self.self_weight)
        pass

    def get_gcn_weight_penalty(self, mode='L2'):

        if mode == 'L1':
            func = lambda x:  t.sum(t.abs(x))
        elif mode == 'L2':
            func  = lambda x: t.sqrt(t.sum(x**2))

        loss = 0 

        tmp = getattr(self.conv1, 'weight', None)
        if tmp is not None: 
            loss += func(tmp)

        tmp = getattr(self.conv1, 'self_weight', None)
        if tmp is not None: 
            loss += 1* func(tmp)

        tmp = getattr(self.conv2, 'weight', None)
        if tmp is not None:
            loss += func(tmp)

        tmp = getattr(self.conv2, 'self_weight', None)
        if tmp is not None:
            loss += 1 * func(tmp)
        # 残差层
        tmp = getattr(self.res_conv1, 'weight', None)
        if tmp is not None:
            loss += func(tmp)

        tmp = getattr(self.res_conv2, 'weight', None)
        if tmp is not None:
            loss += func(tmp)

        tmp = getattr(self.res_conv3, 'weight', None)
        if tmp is not None:
            loss += func(tmp)

        tmp = getattr(self.res_conv4, 'weight', None)
        if tmp is not None:
            loss += func(tmp)

        # 卷积层
        tmp = getattr(self.global_conv1, 'weight', None)
        if tmp is not None: 
            loss += func(tmp)
        tmp = getattr(self.global_conv2, 'weight', None)
        if tmp is not None: 
            loss += func(tmp)

        return loss 


    def forward(self,data,get_latent_varaible=False,ui=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if self.weight_edge_flag:
            one_graph_edge_weight = torch.sigmoid(self.edge_weight)#*self.edge_num

            
            edge_weight = one_graph_edge_weight
        else:
            edge_weight = None 

        x = self.act1(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = help_bn(self.bn1, x)
    



        
        if ui == True:
            c_loss = 0

            c_x = x.permute(1, 0, 2)
            
            avg_pool = F.avg_pool1d(c_x,kernel_size=8)
            #max_pool = F.max_pool1d(c_x, kernel_size=8)
            max_pool = avg_pool.squeeze(dim=-1)
            c_layer = self.c_layer(max_pool)
            c_x = c_x.reshape(c_x.shape[0], -1)
            contrast_label = np.squeeze(data.y, -1).detach()
            contrast_label = torch.tensor(contrast_label).to('cuda')
            contrast_feature = np.squeeze(c_x, -1).detach()
            
            # mean = torch.mean(contrast_feature, dim=0)
         
            # std_dev = torch.std(contrast_feature, dim=0)
            
            # centered_data = contrast_feature - mean
            
            # nor_data = centered_data/(std_dev + 1e-8)
            #
            # cov_matrix = torch.matmul(nor_data.t(), nor_data) /(nor_data.size(0) -1)
            # _, _, pri_components = torch.svd(cov_matrix)
            # k = 63
            # top_pri_conponents = pri_components[:, :k]
            # reduced_data = torch.matmul(nor_data,top_pri_conponents)

            #pca = PCA(n_components=63)
            #representations = pca.fit_transform(contrast_feature.cpu())
            #representations = torch.tensor(representations).to('cuda')
            loss_hard = sup_constrive(c_layer, contrast_label, 0.07)
        else:
            loss_hard = 0




        res = x
        if self.dropout_ratio > 0: x = F.dropout(x, p=0.1, training=self.training)

        #
        x = x.permute(1, 2, 0)
        x = x.unsqueeze(dim=-1)
        #x = x.permute(0, 3, 2, 1)

        h1 = self.res_act1(self.res_conv1(x))
        h1 = self.res_bn1(h1)
        h1 = F.dropout(h1, p=0.3, training=self.training)

        h2 = self.res_act2(self.res_conv2(h1))
        h2 = self.res_bn2(h2)
        h2 = F.dropout(h2, p=0.3, training=self.training)


        h3 = self.res_act3(self.res_conv3(h2))
        h3 = self.res_bn3(h3)
        h3 = F.dropout(h3, p=0.3, training=self.training)


        h4 = h3 + h1
        # res2 = h4
        # h4 = self.res_act2(self.res_conv2(h4))
        # h4 = self.res_bn2(h4)
        # #h4 = F.dropout(h4, p=0.3, training=self.training)
        #
        #
        # h5 = self.res_act3(self.res_conv3(h4))
        # h5 = self.res_bn3(h5)
        # h5 = F.dropout(h5, p=0.3, training=self.training)
        # h6 = h5 + res2

        x = self.res_act4(self.res_conv4(h4))
        x = self.res_bn4(x)
        # x = x.permute(2, 0, 3, 1)+
        # x = x.squeeze(dim=-1)
        x = x.squeeze(dim=-1)
        x = x.permute(2, 0, 1)
        # data_list = [Data(x=x_, edge_index=edge_index) for x_ in x]
        # batch = Batch.from_data_list(data_list)
        # result = self.res_gat(batch.x, edge_index=batch.edge_index)
        x = x + res
        x = self.res_fc(x)
 
        x = self.act2(self.conv2(x, edge_index, edge_weight=edge_weight))
        # # x = res + x
        x = help_bn(self.bn2, x)
        x = F.dropout(x, p=0.3, training=self.training)
        # # #if self.dropout_ratio > 0: x = F.dropout(x, p=0.1, training=self.training)
        #

        x = x.permute(1,2,0)  # #samples x #features x #nodes 
        x = x.unsqueeze(dim=-1) # #samples x #features x #nodes x 1
        x = self.global_conv1(x)  # #samples x #features x #nodes x 1
        x = self.global_act1(x)
        x = self.global_bn1(x)
        if self.dropout_ratio >0: x = F.dropout(x, p=0.3, training=self.training)
        x = self.global_conv2(x)
        x = self.global_act1(x)
        x = self.global_bn2(x)
        if self.dropout_ratio >0: x = F.dropout(x, p=0.3, training=self.training)
        x = x.squeeze(dim=-1)  # #samples  x #features  x #nodes 
        num_samples = x.shape[0]

        x = x .view(num_samples, -1)
        x = self.global_fc_nn(x)
        if get_latent_varaible:
            return x
        else:
            x = self.classifier(x, self.fixed_weight.to(x.device))
            #x = self.fc1(x)
            y = x.cpu().detach().numpy()
            label = data.y.cpu().detach().numpy()
            #label = label.reshape(-1,1)
            #chunk_size = 6784
            lz = np.concatenate((label,y),axis=1)

            with open("oo.csv","a",newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(lz)
                csvfile.close()


            
            return F.softmax(x, dim=-1), loss_hard



def edge_transform_func(org_edge):
    edge = org_edge
    
    edge = t.tensor(edge.T)
    edge = remove_self_loops(edge)[0]
    edge = edge.numpy()
    return edge