#coding:utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
		self.num_layers = 2
		self.emb_size = 200
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

class Model(nn.Module):
	def __init__(self, config):

		super(Model, self).__init__()
		self.n_class = config.num_label
		self.ignore_index = config.ignore_index
		self.num_layers = config.num_layers

		if config.embedding_pretrained is not None:
			self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
		else:
			self.embedding = nn.Embedding(config.vocab_size, config.emb_size,
										  padding_idx=config.vocab_size-1)
			torch.nn.init.uniform_(self.embedding.weight, -0.10, 0.10)

		self.cnn_layer0 = nn.Conv1d(config.emb_size,config.hidden_size,kernel_size=1,padding=0)
		self.cnn_layers = [nn.Conv1d(config.hidden_size,config.hidden_size,kernel_size=3,padding=1) for i in range(config.num_layers-1)]
		for i in range(config.num_layers-1):self.cnn_layers[i]=self.cnn_layers[i].cuda() 
		self.drop = nn.Dropout(config.dropout)
		self.linear = nn.Linear(config.hidden_size, self.n_class)

	def forward(self,input_ids,labels=None):
		emb_out = self.embedding(input_ids).transpose(2,1).contiguous()
		#print(emb_out.size())# = self.embedding(input_ids)
		cnn_output = self.cnn_layer0(emb_out)
		cnn_output = self.drop(cnn_output)
		cnn_output = torch.tanh(cnn_output)
		for i in range(self.num_layers-1):
			cnn_output = self.cnn_layers[i](cnn_output)
			cnn_output = self.drop(cnn_output)
			cnn_output = torch.tanh(cnn_output)
		logits = self.linear(cnn_output.transpose(2,1))
		outputs = (logits,)

		if labels is not None:
			active_logits = logits.view(-1, self.n_class)
			active_labels = labels.view(-1)
			loss = F.cross_entropy(active_logits, active_labels, ignore_index=self.ignore_index)
			outputs = outputs + (loss,)
		return outputs



