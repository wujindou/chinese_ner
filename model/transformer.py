#coding:utf-8
#coding:utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy,math,time
import torch.autograd as autograd

class Config:
    def __init__(self):
        self.model_name = 'cnn'
        # 环境配置
        self.use_cuda = True
        self.device = torch.device('cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu')
        self.device_id = 0
        self.seed = 369

        # 数据配置
        self.data_dir = './data'
        self.do_lower_case = True
        self.label_list = []
        self.num_label = 0
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0

        # logging
        self.logging_dir = './logging/' + self.model_name
        self.visual_log = './v_log/' + self.model_name

        # model
        self.max_seq_length = 64
        self.batch_size = 32
        self.hidden_size = 100
        self.dropout = 0.1
        self.num_layer = 2
        self.emb_size = 200
        self.d_model = 256
        self.d_ff =2*self.d_model
        self.head = 4
        self.use_embedding_pretrained = True
        self.embedding_pretrained_name = 'embedding_Tencent.npz'
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join(self.data_dir, self.embedding_pretrained_name))
            ["embeddings"].astype('float32')) if self.use_embedding_pretrained else None
        self.vocab_size = 0
        self.ignore_index = -100

        # train and eval
        self.learning_rate = 5e-4
        self.weight_decay = 0
        self.num_epochs = 17
        self.early_stop = False
        self.require_improvement = 200
        self.batch_to_out = 50



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)  ## (b,h,l,d) * (b,h,d,l)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn   ##(b,h,l,l) * (b,h,l,d) = (b,h,l,d)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + autograd.Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Model(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, config):
        super(Model, self).__init__()
        self.pad_id = config.pad_id
        self.n_class = config.num_label
        self.ignore_index = config.ignore_index

        c = copy.deepcopy
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.emb_size,
                                          padding_idx=config.vocab_size-1)
            torch.nn.init.uniform_(self.embedding.weight, -0.10, 0.10)
        # attn0 = MultiHeadedAttention(head, d_input, d_model)
        attn = MultiHeadedAttention(config.head, config.d_model, config.dropout)
        ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
        # position = PositionalEncoding(d_model, dropout)
        # layer0 = EncoderLayer(d_model, c(attn0), c(ff), dropout)
        layer = EncoderLayer(config.d_model, c(attn), c(ff), config.dropout)
        self.layers = clones(layer, config.num_layer)
        # layerlist = [copy.deepcopy(layer0),]
        # for _ in range(num_layer-1):
        #     layerlist.append(copy.deepcopy(layer))
        # self.layers = nn.ModuleList(layerlist)
        self.norm = LayerNorm(layer.size)
        self.posi = PositionalEncoding(config.d_model, config.dropout)
        self.input2model = nn.Linear(config.emb_size, config.d_model)
        self.linear = nn.Linear(config.d_model, self.n_class)


    def forward(self, x, labels):
        "Pass the input (and mask) through each layer in turn."
        mask = torch.unsqueeze(1.0-x.eq(self.pad_id).float(),dim=1)
        emb_output= self.embedding(x)
        x = self.posi(self.input2model(emb_output))
        for layer in self.layers:
            x = layer(x,mask=mask)
        out = self.norm(x)
        logits = self.linear(out)
        outputs = (logits,)

        if labels is not None:
            active_logits = logits.view(-1, self.n_class)
            active_labels = labels.view(-1)
            loss = F.cross_entropy(active_logits, active_labels, ignore_index=self.ignore_index)
            outputs = outputs + (loss,)
        return outputs
