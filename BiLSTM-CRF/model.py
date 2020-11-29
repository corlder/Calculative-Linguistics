import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

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
		self.dropout = SpatialDropout(0.6)
		# self.layer_norm = LayerNorm(hidden_size * 2)
		self.classifier = nn.Linear(hidden_size*2,tagset_size)
		self.crf = CRF(tagset_size,batch_first = True)
		# CRF initialization
	
	def forward(self,inputs_ids):
		embs = self.embedding(inputs_ids)
		# embs = self.dropout(embs)
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
