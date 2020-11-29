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

def get_entities_bios(seq,mycvt):
	"""
	Args:
		seq(lsit):sequence of labels
	Returns:
		list:list of (type,start,end)
	"""
	entities = []
	entity = [-1,-1,-1]
	for idx,tag in enumerate(seq):
		if not isinstance(tag,str):
			tag = mycvt.id2label(tag)
		if tag.startswith("S-"):
			if entity[2] != -1:
				entities.append(entity)
			entity = [-1,-1,-1]
			entity[0] = tag.split('-')[1]
			entity[1] = idx
			entity[2] = idx
			entities.append(entity)
			entity = [-1,-1,-1]
		if tag.startswith("B-"):
			if entity[2] != -1:
				entities.append(entity)
			entity = [-1,-1,-1]
			entity[0] = tag.split("-")[1]
			entity[1] = idx
		elif tag.startswith('I-') and entity[1] != -1:
			etype = tag.split('-')[1]
			if etype == entity[0]:
				entity[2] = idx
			if idx == len(seq) - 1:
				entities.append(entity)
		else:
			if entity[2] != -1:
				entities.append(entity)
			entity = [-1,-1,-1]
	return entities
			
def get_entities(seq,mycvt,markup='bios'):
	assert markup in ['bio','bios']
	return get_entities_bios(seq,mycvt)
	
if __name__ == "__main__":
	test = Convertor('./data/vocab.npy','./data/label_voc.npy')
	seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
	print(get_entities(seq,test,'bios'))
	