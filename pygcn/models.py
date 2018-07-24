import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphConvolution_raw, GraphConvolution_GI


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution_raw(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, feat_adj):
        x = F.relu(self.gc1(x, adj, feat_adj))
        # x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, feat_adj)
        return F.log_softmax(x)
