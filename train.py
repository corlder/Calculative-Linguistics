from torch import optim
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
from config import DefaultConfig
from prepare_data import create_vocab,process_data,ClueDataset
from utils import Convertor,get_entities
from model import BiLSTM_CRF
from metrics import Seq2EntityScore

def predict():
	mycvt = Convertor(DefaultConfig.vocab_path,DefaultConfig.label_path)
	mydataset = ClueDataset(os.path.join(DefaultConfig.data_dir,'test.npy'),mycvt)
	clue_dataloader = DataLoader(
		mydataset,
		batch_size = DefaultConfig.batch_size,
		shuffle = False,
		collate_fn = mydataset.collate_fn)
	
	model_path = os.path.join(DefaultConfig.output_dir,"best-model")
	model = BiLSTM_CRF(
		embedding_size = DefaultConfig.embedding_size,
		hidden_size = DefaultConfig.hidden_size,
		drop_out = DefaultConfig.drop_out,
		vocab_size = mycvt.get_vocab_size(),
		target_size = mycvt.get_tagset_size())

	model.load_state_dict(torch.load(model_path))
	results = []
	model.eval()
	metric = Seq2EntityScore(mycvt,'bios')
	
	# id = 0
	with torch.no_grad():
		for idx, batch in enumerate(clue_dataloader):
			input_ids,input_tags,input_lens = batch
			tag_scores = model(input_ids)
			tags_tmp = torch.argmax(tag_scores,dim = 2).tolist()
			tags = [ptag[:ilen] for ptag,ilen in zip(tags_tmp,input_lens)]
			target = [itag[:ilen] for itag,ilen in zip(input_tags.numpy(),input_lens)]
			metric.append(target,tags)
	pred_info,class_info = metric.get_result()
	print(pred_info)
	# print(class_info)
	
		# for line in tag_scores:
			# # not suitable for CRF
			# line = torch.argmax(line,dim=1).tolist()
			# label_entities = get_entities(line,mycvt)
			# json_d = {}
			# json_d['id'] = id
			# json_d['tag_seq'] = " ".join()
			# json_d['entities'] = label_entities
			# results.append(json_d)
			# id += 1
	# test_text = []
	# with open(os.path.join(DefaultConfig.data_dir,'test.json'),'r') as fr:
		# for line in fr:
			# test_text.append(json.loads(line))
	# create_submision()
def train():
	if DefaultConfig.gpu != '':
		device = torch.device(f"cuda:{DefaultConfig.gpu}")
	else:
		device = torch.device("cpu")
	mycvt = Convertor(DefaultConfig.vocab_path,DefaultConfig.label_path)
	mydataset = ClueDataset(os.path.join(DefaultConfig.data_dir,'train.npy'),mycvt)
	clue_dataloader = DataLoader(
		mydataset,
		batch_size = DefaultConfig.batch_size,
		shuffle = True,
		collate_fn = mydataset.collate_fn)
	model = BiLSTM_CRF(
		embedding_size = DefaultConfig.embedding_size,
		hidden_size = DefaultConfig.hidden_size,
		drop_out = DefaultConfig.drop_out,
		vocab_size = mycvt.get_vocab_size(),
		target_size = mycvt.get_tagset_size())
	model.to(device)
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(),lr = DefaultConfig.lr)
	
	for epc in tqdm(range(DefaultConfig.epoch)):
		print("Epoch:%d"%(epc))
		model.train()
		sum = 0
		for idx,batch_samples in tqdm(enumerate(clue_dataloader)):
			input_ids,label_ids,input_lens = batch_samples
			input_ids = input_ids.to(device)
			label_ids = label_ids.to(device)
			tag_scores = model.forward(input_ids)
			# tag_scores,loss = model.forward(input)
			tag_scores = tag_scores.permute(0,2,1)
			loss = loss_function(tag_scores,label_ids)
			# print(loss)
			sum += loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		
		print(sum/336)
		predict()
		model_path = os.path.join(DefaultConfig.output_dir,"best-model")
		torch.save(model.state_dict(),model_path,_use_new_zipfile_serialization=False)
	
	# with torch.no_grad():
		# tag_scores = model(input_test)
		# print(label_test[0])
		# print(tag_scores[0])
	

if __name__ == "__main__":
	torch.set_printoptions(threshold=np.inf)
	create_vocab()
	process_data()
	train()
	predict()
