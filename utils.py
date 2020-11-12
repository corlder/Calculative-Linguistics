import numpy as np

class Convertor(object):
	
	def __init__(self,vocab_path,label_voc_path):
		self.vocab = np.load(vocab_path,allow_pickle=True).item()
		self.label_voc = np.load(label_voc_path,allow_pickle=True).item()
		self.rev_vocab = dict(zip(self.vocab.values(),self.vocab.keys()))
		self.rev_label_voc = dict(zip(self.label_voc.values(),self.label_voc.keys()))
	
	def word2id(word):
		return self.vocab[word]
		
	def id2word(id):
		return self.rev_vocab[id]
	
	def label2id(label):
		return self.label_voc[label]
		
	def id2label(id):
		return self.rev_label_voc[id]
