from torch import optim
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import DefaultConfig
from prepare_data import create_vocab,process_data,ClueDataset
from utils import Convertor
from model import BiLSTM_CRF


if __name__ == "__main__":
	create_vocab()
	process_data()
	
	test_input = torch.LongTensor([[1642, 1291,   40, 2255,  970,   46,  124, 1604, 1915,  547,    0,  173,
          303,  124, 1029,   52,   20, 2839,    2, 2255, 2078, 1553,  225,  540,
           96,  469, 1704,    0,  174,    3,    8,  728,  903,  403,  538,  668,
          179,   27,   78,  292,    7,  134, 2078, 1029,    0,    0,    0,    0,
            0],
        [  28,    6,  926,   72,  209,  330,  308,  167,   87, 1345,    1,  528,
          412,    0,  584,    1,    6,   28,  326,    1,  361,  342, 3256,   17,
           19, 1549, 3257,  131,    2,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0],
        [   6,    3,   58, 1930,   37,  407, 1068,   40, 1299, 1443,  103, 1235,
         1040,  139,  879,   11,  124,  200,  135,   97, 1138, 1016,  402,  696,
          337,  215,  402,  288,   10,    5,    5,   17,    0,  248,  597,  110,
           84,    1,  135,   97, 1138, 1016,  402,  696,  402,  200,  109,  164,
            0],
        [ 174,    6,  110,   84,    3,  477,  332,  133,   66,   11,  557,  107,
          181,  350,    0,   70,  196,  166,   50,  120,   26,   89,   66,   19,
          564,    0,   36,   26,   48,  243, 1308,    0,  139,  212,  621,  300,
            0,  444,  720,    4,  177,  165,  164,    2,    0,    0,    0,    0,
            0]])
	test_label = torch.LongTensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4, 14, 14, 14, 14, 14,
         14, 14, 14, 14, 14,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  4, 14, 14, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  1, 11,  0,  1, 11, 11, 11, 11, 11,  0,  0,  0,  0,  8, 18, 18,
         18, 18, 18, 18, 18, 18,  0,  0,  9, 19,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  8, 18, 18, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8, 18,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
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
	loss_function = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(),lr = DefaultConfig.lr)
	with torch.no_grad():
		tag_scores = model(test_input)
		print(tag_scores)
		print(tag_scores.size())
	for idx,batch_samples in enumerate(clue_dataloader):
		# print(idx, batch_samples)
		if idx >= 100:
			break
		input_ids,label_ids,input_lens = batch_samples
		model.zero_grad()
		tag_scores = model.forward(input_ids)
		tag_scores = tag_scores.permute(0,2,1)
		loss = loss_function(tag_scores,label_ids)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		# print(idx)
	print("trainning end")
	with torch.no_grad():
		tag_scores = model(test_input)
		print(test_label)
		print(tag_scores)
