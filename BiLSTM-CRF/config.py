
class DefaultConfig(object):

	vocab_size = 100000
    
	data_dir = '../data/'
	output_dir = './checkpoints/'
	vocab_path = '../data/vocab.npy'
	label_path = '../data/label_voc.npy'
	
	batch_size = 32
	epoch = 15		# to be modified
	embedding_size = 128
	hidden_size = 384
	drop_out = 0.5
	max_epoch = 10
	lr = 0.001
	lr_decay = 0.95

	gpu = '2'
	
    
