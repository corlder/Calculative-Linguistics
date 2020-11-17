import torch
from utils import get_entities
from collections import Counter

class Seq2EntityScore(object):
	def __init__(self,convertor,mode='bios'):
		self.mycvt = convertor
		self.mode = mode
		self.reset()
	
	def reset(self):
		self.origins = []
		self.founds = []
		self.rights = []
		
	def compute(self,origin,found,right):
		recall = 0 if origin == 0 else (right / origin)
		precision = 0 if found == 0 else (right / found)
		f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
		return recall, precision, f1
		
	def append(self,label_seqs,pred_seqs):
		'''
		CRF can guarantee that the entities exceeding the length of the orgin seq not existing.
		'''
		for label_seq, pred_seq in zip(label_seqs,pred_seqs):
			label_entities = get_entities(label_seq,self.mycvt,self.mode)
			pred_entities = get_entities(pred_seq,self.mycvt,self.mode)
			self.origins += label_entities
			self.founds += pred_entities
			self.rights += [pred_entity for pred_entity in pred_entities if pred_entity in label_entities]
	
	def get_result(self):
		class_info = {}
		origin_counter = Counter([x[0] for x in self.origins])
		found_counter = Counter([x[0] for x in self.founds])
		right_counter = Counter([x[0] for x in self.rights])
		for etype, count in origin_counter.items():
			origin = count
			found = found_counter.get(etype,0)
			right = right_counter.get(etype,0)
			recall,precision,f1 = self.compute(origin,found,right)
			class_info[etype] = {"acc": round(precision,4),"recall":round(recall,4),"f1":round(f1,4)}
		origin = len(self.origins)
		found = len(self.founds)
		right = len(self.rights)
		recall,precision,f1 = self.compute(origin,found,right)
		return {"acc":precision,"recall":recall,"f1":f1},class_info