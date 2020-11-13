from torch import optim
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from config import DefaultConfig
from prepare_data import create_vocab,process_data,ClueDataset
from utils import Convertor

if __name__ == "__main__":
	create_vocab()
	process_data()
	
	mycvt = Convertor(DefaultConfig.vocab_path,DefaultConfig.label_path)
	mydataset = ClueDataset(os.path.join(DefaultConfig.data_dir,'train.npy'),mycvt)
	clue_dataloader = DataLoader(
		mydataset,
		batch_size = DefaultConfig.batch_size,
		shuffle = True,
		collate_fn = mydataset.collate_fn
		)
		
	for idx,batch_samples in enumerate(clue_dataloader):
		print(idx, batch_samples)
		if idx >= 3:
			break
	
