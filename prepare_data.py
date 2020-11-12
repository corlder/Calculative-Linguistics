from config import DefaultConfig
import numpy as np
import json
import os

def prepare_for_data():
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
	sorted(word_freq.items(),key=lambda e:e[1],reverse=True)
	for word,cnt in word_freq.items():
		vocab[word] = i
		i += 1
		if i>=max_size:
			break
	np.save(vocab_path,vocab)
		
