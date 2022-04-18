import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import gumbel_softmax


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_x):
        # Linear Transformation
        h = torch.mm(input_x, self.W)
        N = h.size()[0]

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 直接return e 可不可以？？？
        # Masked Attention
        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(e > 0.5, e, zero_vec)
        attention = e
        # 全连接的注意力机制
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, dropout, alpha, n_heads):
        """
           参数1 ：nfeat   输入层数量
           参数2： nhid    输出特征数量
           参数3： nclass  分类个数
           参数4： dropout dropout 斜率
           参数5： alpha  激活函数的斜率
           参数6： nheads 多头部分

        """
        super(GAT, self).__init__()
        self.dropout = dropout
        # 根据多头部分给定的数量声明attention的数量
        self.attentions = [GATLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        # 将多头的各个attention作为子模块添加到当前模块中。
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 最后一个attention层，输出的是分类
        # self.out_att = GATLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    # 前向传播过程
    def forward(self, x):
        # 参数x：各个输入的节点得特征表示
        # 参数adj：邻接矩阵表示
        x = F.dropout(x, self.dropout, training=self.training)
        # 对每一个attention的输出做拼接

        # 需不需要将x归一化呢？
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # self.out_att(x,adj)
        # 输出的是带有权重的分类特征
        # x = F.elu(self.out_att(x, adj))
        # 各个分类的概率归一化
        return x


class GraphLearning(nn.Module):
    def __init__(self, n_in, n_in_reversed, n_hid, n_out, n_out_reserved, node_labels, num_region, gat_dropout,
                 num_heads, mlp_dropout=0.5):
        super(GraphLearning, self).__init__()
        self.in_len = n_in
        self.reversed_len = n_in_reversed
        self.mlp_dropout = mlp_dropout
        self.final_dim = num_region * n_hid
        # self.f_1 = MLP(n_in, n_hid, n_hid, mlp_dropout)
        self.f_1 = MLP(n_in, n_hid, n_hid, mlp_dropout)
        self.gat = GAT(n_hid, n_hid=n_hid, dropout=gat_dropout, alpha=0.2, n_heads=num_heads)
        self.f_2 = MLP(n_hid * 8, n_hid, 1, mlp_dropout)
        # self.f_3 = MLP(n_in, n_hid, n_hid, mlp_dropout)
        # self.f_4 = nn.Linear(node_labels, n_out_reserved)
        # self.f_5 = MLP(2 * n_out_reserved, n_hid, n_hid, mlp_dropout)
        self.f_6 = MLP(n_hid*2, n_hid, n_hid, mlp_dropout)
        # self.f_7_1 = MLP(n_hid*2, n_hid, n_hid)
        self.f_7_2 = nn.Linear(self.final_dim, n_out)

        # self.mlp4 = MLP(n_hid + 64, n_hid, n_hid, do_prob)
        # self.mlp3 = nn.Linear(3, 32)
        # self.bn = nn.BatchNorm1d(n_hid * 3)

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs, node_class, receive, send,temp):
        x = inputs.to(torch.float32)
        # print(x)
        # print(x)
        x_1 = x
        # print(x_1.shape)
        x_1 = self.f_1(x_1)
        x_1_gat = torch.stack([self.gat(x_1[i, :, :]) for i in range(x_1.size(0))], 0)
        x_1 = torch.cat([x_1, x_1_gat], dim=-1)
        # # print(x_1.shape)
        # # print(x_1)
        # print(x_1[...,0:5])
        #  错了啊兄弟
        receivers = torch.matmul(receive, x_1)
        # print(receivers[0:20,0:5])
        senders = torch.matmul(send, x_1)
        x_1 = torch.cat([senders, receivers], dim=2)
        # print(x_1[...,0:5])





        # # print(x_1.shape)
        x_1 = self.f_2(x_1)
        x_ij = gumbel_softmax(x_1, temp)
        #
        x_2 = x
        # # print(x_2.shape)
        x_2 = self.f_1(x_2)
        # # node_class = node_class.to(torch.float32)
        # # x_prior = self.f_4(node_class)
        # # x_piror_cat = torch.stack([x_prior for i in range(x_2.size(0))], dim=0)
        # # x_i_cat = torch.cat((x_2, x_piror_cat), dim=2)
        # # x_add_piror = self.f_5(x_i_cat)
        # # receivers_2 = torch.matmul(receive, x_add_piror)
        # # senders_2 = torch.matmul(send, x_add_piror)
        receivers_2 = torch.matmul(receive, x_2)
        senders_2 = torch.matmul(send, x_2)
        x_concat = torch.cat([senders_2, receivers_2], dim=-1)
        # # print(x_concat.shape)
        x_concat = self.f_6(x_concat)
        x_concat_t = torch.transpose(x_concat, 1, 2)
        x_ij = torch.transpose(x_ij, 1, 2)
        # print(x_concat_t.shape, x_ij.shape)
        z = torch.mul(x_concat_t, x_ij)
        # print(z.shape)
        z = torch.transpose(z, 1, 2).contiguous()
        # print(z.shape)
        z = z.transpose(-2, -1).matmul(send).transpose(-2, -1)
        z = z.contiguous()
        z = torch.cat([x_2, z], dim=-1)
        # print('\n',z.shape)
        # z = z.view(-1,self.final_dim)
        # # y = F.dropout(F.relu(self.f_7_1(z)), p=self.mlp_dropout)
        # # y = self.f_7_2(y)
        z = self.f_6(z)
        # print(z.shape)
        # print(z.shape)
        z = z.view(z.size(0), 1, z.size(1) * z.size(2))
        z = torch.squeeze(z, dim=1)
        # print(z.shape)
        y = self.f_7_2(z)
        # print(y.shape)
        return y
