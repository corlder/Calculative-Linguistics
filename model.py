import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
	
	def __init__(self,embedding_size,hidden_size,vocab_size,tagset_size,drop_out):
		super(BiLSTM_CRF,self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size,embedding_size)
		self.bilstm = nn.LSTM(
			input_size = embedding_size,
			hidden_size = hidden_size,
			batch_first = True,
			num_layers = 2,
			dropout = drop_out,
			bidirectional = True
		)
		# self.layer_norm = LayerNorm(hidden_size * 2)
		self.classifier = nn.Linear(hidden_size*2,tagset_size)
		self.crf = CRF(tagset_size,batch_first = True)
		# CRF initialization
	
	def forward(self,inputs_ids):
		embs = self.embedding(inputs_ids)
		# while not pass the (h0,c0), the bilstm will be initialized
		sequence_output, _ = self.bilstm(embs)
		# sequence_output = self.layer_norm(sequence_output)
		features = self.classifier(sequence_output)
		return features
		# tag_scores = F.log_softmax(features,dim=2)
		# return tag_scores
	
	def forward_get_loss(self,input_ids,input_mask,input_lens,input_tags):
		tag_scores = self.forward(input_ids)
		loss = self.crf(tag_scores,input_tags,input_mask) * (-1)
		return tag_scores,loss
