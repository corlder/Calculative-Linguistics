import numpy as np

class Convertor(object):
	
	def __init__(self,vocab_path,label_voc_path):
		self.vocab = np.load(vocab_path,allow_pickle=True).item()
		self.label_voc = np.load(label_voc_path,allow_pickle=True).item()
		self.rev_vocab = dict(zip(self.vocab.values(),self.vocab.keys()))
		self.rev_label_voc = dict(zip(self.label_voc.values(),self.label_voc.keys()))
	
	def word2id(self,word):
		return self.vocab[word]
		
	def id2word(self,id):
		return self.rev_vocab[id]
	
	def label2id(self,label):
		return self.label_voc[label]
		
	def id2label(self,id):
		return self.rev_label_voc[id]
		
	def get_vocab_size(self):
		return len(self.vocab)
	
	def get_tagset_size(self):
		return len(self.label_voc)
	

if __name__ == "__main__":
	test = Convertor('./data/vocab.npy','./data/label_voc.npy')
	print(test.id2word(1))