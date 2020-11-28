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
import sys

def predict():
	if DefaultConfig.gpu != '':
		device = torch.device(f"cuda:{DefaultConfig.gpu}")
	else:
		device = torch.device("cpu")
		
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
		tagset_size = mycvt.get_tagset_size())

	model.load_state_dict(torch.load(model_path))
	model.to(device)
	results = []
	model.eval()
	metric = Seq2EntityScore(mycvt,'bios')
	case = []
	# id = 0
	with torch.no_grad():
		for idx, batch in enumerate(clue_dataloader):
			input_ids,input_tags,mask,input_lens = batch
			input_ids = input_ids.to(device)
			input_tags = input_tags.to(device)
			mask = mask.to(device)
			tag_scores = model.forward(input_ids)
			tags = model.crf.decode(tag_scores,mask=mask)
			# tags_tmp = torch.argmax(tag_scores,dim = 2).tolist()
			# tags = [ptag[:ilen] for ptag,ilen in zip(tags_tmp,input_lens)]
			target = [itag[:ilen] for itag,ilen in zip(input_tags.cpu().numpy(),input_lens)]
			tags_cvted = [[mycvt.id2label(x) for x in line]for line in tags]
			target_cvted = [[mycvt.id2label(x) for x in line] for line in target]
			case.extend(list(zip(tags_cvted,target_cvted)))
			metric.append(target,tags)
			
	pred_info,class_info = metric.get_result()
	print("-----------------testing result------------------")
	for k,v in class_info.items():
		print("[test] %s: acc:%.4f, recall:%.4f, f1:%.4f"%(k,v['acc'],v['recall'],v['f1']))
		
	# print(class_info)
	print("[test] total acc:%.4f, recall:%.4f, f1:%.4f"%(pred_info['acc'],pred_info['recall'],pred_info['f1']))
	# print(class_info)
	np.save("raw_case.npy",case)
	sys.stdout.flush()
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
		tagset_size = mycvt.get_tagset_size())
	model.to(device)
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(),lr = DefaultConfig.lr)
	# how to initialize these parameters elegantly
	for p in model.crf.parameters():
		_ = torch.nn.init.uniform(p,-1,1)
	
	for epc in range(DefaultConfig.epoch):
		print("Epoch:%d"%(epc))
		model.train()
		s = 0
		for idx,batch_samples in enumerate(clue_dataloader):
			input_ids,label_ids,mask,input_lens = batch_samples
			input_ids = input_ids.to(device)
			mask = mask.to(device)
			label_ids = label_ids.to(device)
			#tag_scores = model.forward(input_ids)
			tag_scores,loss = model.forward_get_loss(input_ids,mask,input_lens,label_ids)
			# tag_scores = tag_scores.permute(0,2,1)
			# loss = loss_function(tag_scores,label_ids)
			# print(loss)
			s += loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		
		print("training loss:",(float(s.data))/336)
		model_path = os.path.join(DefaultConfig.output_dir,"best-model")
		torch.save(model.state_dict(),model_path,_use_new_zipfile_serialization=False)
		# print("--------------------best-model saved-----------------------")
		predict()
		
	# with torch.no_grad():
		# tag_scores = model(input_test)
		# print(label_test[0])
		# print(tag_scores[0])
	sys.stdout.flush()

if __name__ == "__main__":
	torch.set_printoptions(threshold=np.inf)
	create_vocab()
	process_data()
	train()
	predict()
