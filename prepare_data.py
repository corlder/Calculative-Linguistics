from config import DefaultConfig
from torch.utils.data import Dataset
from utils import Convertor
import numpy as np
import json
import os

from config import DefaultConfig
from torch.utils.data import Dataset,DataLoader
from utils import Convertor
import torch
import numpy as np
import json
import os

class ClueDataset(Dataset):

    def __init__(self,file_path,convertor):
        self.mycvt = convertor
        self.dataset = self.convert2id(np.load(file_path,allow_pickle=True))
        
    def convert2id(self,raw_list):
        
        prd_list = []
        for x in raw_list:
            text_ids = [self.mycvt.word2id(i) for i in x[0]]
            tag_ids = [self.mycvt.label2id(i) for i in x[1]]
            prd_list.append([text_ids,tag_ids])
        return prd_list

    def __len__(self):
        return len(self.dataset)
    
    def get_long_tensor(self,tokens_list,batch_size):
        
        token_len = max([len(x) for x in tokens_list])
        tokens = torch.LongTensor(batch_size,token_len).fill_(0)
        for i,s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
        return tokens
    
    def collate_fn(self,batch):
        
        texts = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x) for x in texts]
        batch_size = len(batch)

        input_ids = self.get_long_tensor(texts,batch_size)
        label_ids = self.get_long_tensor(labels,batch_size)
        
        return [input_ids,label_ids,lens]

    def __getitem__(self,idx):
        text = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [text,label]
		

def process_data():
	'''
	'''
	files = ["train","test"]
	for file in files:
		input_path = os.path.join(DefaultConfig.data_dir,file + '.json')
		get_examples(input_path,file)
	


def get_examples(input_path,mode):
	'''
	'''
	data_dir = os.path.dirname(input_path)
	output_path = os.path.join(data_dir,str(mode) + ".npy")
	
	if os.path.exists(output_path) is True:
		return
	with open(input_path,"r",encoding="utf-8") as fr:
		res = []
		for line in fr:
			line = json.loads(line.strip())
			
			text = line['text']
			label_entities = line.get('label',None)
			words = list(text)
			labels = ['O'] * len(words)
			
			if label_entities is not None:
				for key,value in label_entities.items():
					for sub_name,sub_index in value.items():
						for start_index,end_index in sub_index:
							assert ''.join(words[start_index:end_index + 1]) == sub_name
							if start_index == end_index:
								labels[start_index] = 'S-' + key
							else:
								labels[start_index] = 'B-' + key
								labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
			res.append((words,labels))
		np.save(output_path,res)
	
def create_vocab():
	'''
	'''
	vocab_path = DefaultConfig.vocab_path
	data_dir = DefaultConfig.data_dir
	files = ["train.json","test.json"]
	
	if os.path.exists(vocab_path):
		return
	word_freq = {}
	max_size = DefaultConfig.vocab_size
	for file in files:
		with open(os.path.join(data_dir,file),'r') as fr:
			for line in fr:
				line = json.loads(line.strip())
				text = line['text']
				text = list(text)
				for ch in text:
					if ch in word_freq:
						word_freq[ch] += 1
					else:
						word_freq[ch] = 1	
	i = 0
	vocab = {}
	tmp = sorted(word_freq.items(),key=lambda e:e[1],reverse=True)
	for elem in tmp:
		vocab[elem[0]] = i
		i += 1
		if i>=max_size:
			break
	np.save(vocab_path,vocab)
		
