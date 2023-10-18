"""
Main function of GCN-based prediction of SC-FC relationships in early childhood.
"""
import torch
# torch.use_deterministic_algorithms(True)
torch.set_deterministic(True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import random
from model import *
from train import *
import argparse
from sklearn.model_selection import StratifiedKFold, KFold
import networkx as nx
import itertools

# from data_preprocessing_coupling_FD import *
from scipy.linalg import eigvals

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='GCN_SC_FC')
	parser.add_argument('--epochs', type=int, default=100, metavar='no_epochs',
				help='number of episode to train ')
	parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
				help='learning rate (default: 0.0001 using Adam Optimizer)')
	parser.add_argument('--beta1', type=float, default=0.9, 
				help='coefficients used for computing running averages of gradient (default: 0.9 using Adam Optimizer)')
	parser.add_argument('--weight_decay', type=float, default=5e-4,
					help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--gamma', type=float, default=0.95,
					help='decay for learning_scheduler')
	parser.add_argument('--lam_class', type=float, default=0.1, 
				help='classification error hyperparameter')
	parser.add_argument('--lam_adv', type=float, default=0.01, 
				help='Adversarial loss hyperparameter')
	parser.add_argument('--lam_reg', type=float, default=1.0, 
				help='Regularization hyperparameter')
	parser.add_argument('--lam_corr', type=float, default=1.5, 
				help='Correlation error hyperparameter')
	parser.add_argument('--lam_siamese', type=float, default=1.0, 
				help='Regularization hyperparameter')
	parser.add_argument('--reg_gamma', type=float, default=0.45, 
				help='Regularization hyperparameter')
	parser.add_argument('--lr_dim', type=int, default=90, metavar='N',
				help='adjacency matrix input dimensions')
	parser.add_argument('--hr_dim', type=int, default=90, metavar='N',
				help='super-resolved adjacency matrix output dimensions')
	parser.add_argument('--batch_size', type=int, default=1, metavar='N',
				help='batch sizes')
	parser.add_argument('--log_name', type=str, default='SCtoFC_chebGraphConv_5hops_layernorm_group_p.01', 
				help='file name to save errors')
	parser.add_argument('--optimizer', type=str, default='ADAM', 
				help='optimizer')
	parser.add_argument('--log_path', type=str, default='./Results/', 
				help='file name to save errors')
	parser.add_argument('--save_dir', type=str, default='./Checkpoints/', 
				help='file name to save errors')
	parser.add_argument('--methods_thresh', type=str, default='All_aal_2yrspace_ROI_FC_Z_matrix_group_p.01_uncorrected', 
				help='different thresholding with individual/group level')
	parser.add_argument('--SCyear', type=int, default=6, metavar='N',
				help='age for SC')
	parser.add_argument('--FCyear', type=int, default=6, metavar='N',
				help='age for FC')
	args = parser.parse_args()

	def calc_chebynet_gso(gso):
		id = np.identity(gso.shape[0])
		eigval_max = max(eigvals(a=gso).real)
		
		# If the gso is symmetric or random walk normalized Laplacian,
		# then the maximum eigenvalue has to be smaller than or equal to 2.
		if eigval_max >= 2:
			gso = gso - id
		else:
			gso = 2 * gso / eigval_max - id

		return gso
	
	def normalize(mx):
		"""Row-normalize sparse matrix"""
		rowsum = np.array(mx.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.
		r_mat_inv = sp.diags(r_inv)
		mx = r_mat_inv.dot(mx)
		return mx
	
	# edge_SC1, edge_FC1, subjects_ids1, _, scanner1, _, _, _, _, _ = data_EBDS_1yr_SC_FC(args.methods_thresh, args.lr_dim) 
	# edge_SC2, edge_FC2, subjects_ids2, _, scanner2, _, _, _, _, _ = data_EBDS_2yr_SC_FC(args.methods_thresh, args.lr_dim) 
	# edge_SC4, edge_FC4, subjects_ids4, _, scanner4, _, _, _, _, _ = data_EBDS_4yr_SC_FC(args.methods_thresh, args.lr_dim) 
	# edge_SC6, edge_FC6, subjects_ids6, _, scanner6, _, _, _, _, _ = data_EBDS_6yr_SC_FC(args.methods_thresh, args.lr_dim) 
	
	# edge_SC = edge_SC1 + edge_SC2 + edge_SC4 + edge_SC6
	# edge_FC = edge_FC1 + edge_FC2 + edge_FC4 + edge_FC6
	# subjects_ids = subjects_ids1 + subjects_ids2 + subjects_ids4 + subjects_ids6
	# scanners = scanner1 + scanner2 + scanner4 + scanner6

	# edge_SC = np.array(edge_SC)
	# edge_FC = np.array(edge_FC)

	## define SC and FC, subjects ids, scanners, ages
	edge_SC = np.random.randn(40, args.lr_dim, args.lr_dim)
	edge_FC = np.random.randn(40, args.lr_dim, args.lr_dim)
	subjects_ids = np.arange(40)
	scanners = 


	# rescaling individual level
	for s in range(edge_SC.shape[0]):
		content = edge_SC[s]
		
		# row-wise normalization
		content = normalize(content)
		# Kipf: renomalization trick
		content = content + np.identity(content.shape[0])
		content = normalize(content)
		# Laplacian
		content = np.identity(content.shape[0]) - content
		# Laplacian rescale
		content = calc_chebynet_gso(content)
		
		edge_SC[s] = content
	
	maxval = np.arctanh(0.99)#np.percentile(edge_FC, 99)#.99)
	edge_FC = (edge_FC-0)/(maxval-0)
	edge_FC[edge_FC>1.0] = 1.0
	edge_FC[edge_FC<0.0] = 0.0


	mean_adj = np.mean(edge_FC, 0)*maxval
	mean_adj = mean_adj
	filedir = os.path.join(args.log_path, args.log_name)
	if not os.path.exists(os.path.join(args.log_path, args.log_name)):
		os.makedirs(os.path.join(args.log_path, args.log_name))
	filename = filedir +'/'+'meanFC.txt'	
	with open(filename, "w") as log_file:
		np.savetxt(log_file, mean_adj, fmt='%.7f')


	
	num_CV = 0

	skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
	idx = np.linspace(0,len(subjects_ids)-1,len(subjects_ids),dtype=int)

	age_group = np.zeros(len(subjects_ids), dtype=np.int32)
	print ("Total number of samples: ", len(subjects_ids))
	cnt0, cnt1, cnt2, cnt3 = 0,0,0,0
	for s in range(len(subjects_ids)):
		if subjects_ids[s].endswith('1year'):
			age_group[s] = 0
			cnt0+=1
		elif subjects_ids[s].endswith('2year'):
			age_group[s] = 1
			cnt1+=1
		elif subjects_ids[s].endswith('4year'):
			age_group[s] = 2   
			cnt2+=1
		elif subjects_ids[s].endswith('6year'):
			age_group[s] = 3 
			cnt3+=1
	print ("1 year samples: ", cnt0)
	print ("2 year samples: ", cnt1)
	print ("4 year samples: ", cnt2)
	print ("6 year samples: ", cnt3)

	age_scanner_group = np.zeros(len(subjects_ids), dtype=np.int32)
	cnt0, cnt1, cnt2, cnt3, cnt4, cnt5, cnt6, cnt7 = 0,0,0,0,0,0,0,0
	for s in range(len(subjects_ids)):
		if subjects_ids[s].endswith('1year'):
			if scanners[s]=='A':
				age_scanner_group[s] = 0
				cnt0+=1
			else:
				age_scanner_group[s] = 1
				cnt1+=1
		elif subjects_ids[s].endswith('2year'):
			if scanners[s]=='A':
				age_scanner_group[s] = 2
				cnt2+=1
			else:
				age_scanner_group[s] = 3
				cnt3+=1
		elif subjects_ids[s].endswith('4year'):
			if scanners[s]=='A':
				age_scanner_group[s] = 4
				cnt4+=1
			else:
				age_scanner_group[s] = 5
				cnt5+=1
		elif subjects_ids[s].endswith('6year'):
			if scanners[s]=='A':
				age_scanner_group[s] = 6
				cnt6+=1
			else:
				age_scanner_group[s] = 7
				cnt7+=1
	print ("1 year Allergra: ", cnt0)
	print ("1 year Trio: ", cnt1)
	print ("2 year Allergra: ", cnt2)
	print ("2 year Trio: ", cnt3)
	print ("4 year Allergra: ", cnt4)
	print ("4 year Trio: ", cnt5)
	print ("6 year Allergra: ", cnt6)
	print ("6 year Trio: ", cnt7)
	num_CV = 0
	for train_index, test_index in skf.split(idx, age_scanner_group):

		lrs =  [args.lr]
		stepsizes = [10]#, 20, 50]#, 300]
		params = list(itertools.product(stepsizes, lrs))	

		# Empty lists for storing results
		error_store, accuracy_store = [], []

		# Grid search
		for stepsizes, lrs  in params:
			print ("lr: ", lrs, "stepsize: ", stepsizes)
			# print ("Test idx: ", test_index)
			print ("Test subjects:", [subjects_ids[index] for index in test_index])
			training_adj, test_adj = [edge_SC[index] for index in train_index],  [edge_SC[index] for index in test_index]
			training_labels, test_labels = [edge_FC[index] for index in train_index],  [edge_FC[index] for index in test_index]
			training_ids, test_ids = [subjects_ids[index] for index in train_index], [subjects_ids[index] for index in test_index]
			training_age, test_age = [age_group[index] for index in train_index], [age_group[index] for index in test_index]
			training_scanner, test_scanner = [scanners[index] for index in train_index], [scanners[index] for index in test_index]

			training_adj = list(training_adj)
			training_labels = list(training_labels)
			training_ids = list(training_ids)
			training_age = list(training_age)
			training_scanner = list(training_scanner)

			test_adj = list(test_adj)
			test_labels = list(test_labels)
			test_ids = list(test_ids)
			test_age = list(test_age)
			test_scanner = list(test_scanner)


			torch.manual_seed(0)
			torch.cuda.manual_seed(0)
			torch.cuda.manual_seed_all(0)
			os.environ["PYTHONHASHSEED"] = str(0)
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
			random.seed(0)
			np.random.seed(0)
			model = SC_FC_age_ordinal_scanner_GraphConv_siamese(args)
			if torch.cuda.is_available():
				device = 'cuda'
				model.to(device)


			test_mae = train(model, training_adj, training_labels, training_ids, training_age, training_scanner, 
					test_adj, test_labels, test_ids, test_age, test_scanner, num_CV, 0, lrs, stepsizes, maxval,np.mean(edge_FC, 0),args)
			print ("lr: ", lrs, "stepsizes: ", stepsizes, "test error: ", test_mae)
			# Store grid search results
			error_store.append(test_mae)
		# Determine the best parameters
		print (error_store)
		
		
		num_CV +=1
		# break

