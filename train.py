from torch import optim
import torch
import torch.nn as nn
from config import DefaultConfig
from prepare_data import create_vocab,process_data
from utils import Convertor

if __name__ == "__main__":
	mycvt = Convertor(DefaultConfig.vocab_path,DefaultConfig.label_path)
	
