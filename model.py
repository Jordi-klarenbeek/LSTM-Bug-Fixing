import random
import util
import time
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeqEncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers, bidirect):
        super(SeqEncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        if bidirect:
            self.direct = 2
        else:
            self.direct = 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size // self.direct, num_layers, batch_first=True, bidirectional=bidirect)

    def forward(self, input, hidden_cell):
        input_lengths = util.list_lengths(input)
        output = self.embedding(input).view(self.batch_size, input.shape[1], -1)
        packed_output = pack_padded_sequence(output, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden_cell = self.lstm(packed_output, hidden_cell)
        return output, hidden_cell

    # Initialize the hidden and cell state
    def initHidden(self):
        return (torch.rand(self.num_layers*self.direct, self.batch_size, self.hidden_size // self.direct, device=device),
                torch.rand(self.num_layers*self.direct, self.batch_size, self.hidden_size // self.direct, device=device))

# Child-sum treelstm from unbounce/
class ChildSumTreeLSTM(torch.nn.Module):
    '''PyTorch TreeLSTM model that implements efficient batching.
    '''
    def __init__(self, vocab_size, hidden_size):
        '''TreeLSTM class initializer
        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.U_iou = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)

        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.U_f = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, features, node_order, adjacency_list, edge_order):
        '''Run TreeLSTM model on a tree data structure with node features
        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        '''

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        # populate the h and c states respecting computation order
        for n in range(node_order.max() + 1):
            self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order)

        return (h, c)

    def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        '''Helper function to evaluate all tree nodes currently able to be evaluated.
        '''
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size n x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size e x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]

        # pass the features through the embedding layer to get the token embeddings
        embed_x = torch.squeeze(self.embedding(x))

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(embed_x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            h_sum = torch.stack(parent_list)
            iou = self.W_iou(embed_x) + self.U_iou(h_sum)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # embedded features of parents
            parent_embeddings = torch.squeeze(self.embedding(features[parent_indexes, :]))

            # f is a tensor of size e x M
            f = self.W_f(parent_embeddings) + self.U_f(child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)

            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])

class SeqDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, num_layers):
        super(SeqDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_cell, encoder_outputs):
        output = self.embedding(input).view(1, self.batch_size, -1)
        output = F.relu(output)
        output, hidden_cell = self.lstm(output, hidden_cell)
        output = self.out(output[0,:])
        #self.softmax(
        return output, hidden_cell

    def initHidden(self):
        return (torch.rand(self.num_layers, self.batch_size, self.hidden_size, device=device),
                torch.rand(self.num_layers, self.batch_size, self.hidden_size, device=device))

class SeqDecoderAttentionLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, max_length, num_layers, dropout_p):
        super(SeqDecoderAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_cell, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        embedded = self.dropout(embedded)

        cat = torch.cat((embedded[0], hidden_cell[0][0]), 1)
        attn_weights = F.softmax(self.attn(cat), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).view(1,self.batch_size,self.hidden_size)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden_cell = self.lstm(output, hidden_cell)

        output = F.log_softmax(self.out(output[0, :]), dim=1)
        return output, hidden_cell

    def initHidden(self):
        return (torch.rand(self.num_layers, self.batch_size, self.hidden_size, device=device),
                torch.rand(self.num_layers, self.batch_size, self.hidden_size, device=device))

class BinaryClassifierNet(nn.Module):
    def __init__(self, hidden_size, num_layers_decoder):
        super(BinaryClassifierNet,self).__init__()
        self.fc1 = nn.Linear(hidden_size*num_layers_decoder,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = torch.sigmoid(self.fc3(x))
        return out