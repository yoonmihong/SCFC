import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
# from preprocessing import *
from model import *
import torch.optim as optim
# from collections import OrderedDict
# import csv
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from pytorchtools import EarlyStopping
import os
from sklearn.model_selection import train_test_split
import pandas as pd
# import resreg
import networkx as nx
from torch.utils.tensorboard import SummaryWriter
# Writer will output to ./runs/ directory by default

class FocalLoss(nn.Module):
	
	def __init__(self, weight=None, 
				 gamma=2., reduction='none'):
		nn.Module.__init__(self)
		self.weight = weight
		self.gamma = gamma
		self.reduction = reduction
		
	def forward(self, input_tensor, target_tensor):
		# log_prob = F.log_softmax(input_tensor, dim=-1)
		log_prob = input_tensor
		prob = torch.exp(log_prob)
		return F.nll_loss(
			((1 - prob) ** self.gamma) * log_prob, 
			target_tensor, 
			weight=self.weight,
			reduction = self.reduction
		).mean()
criterion_focalloss = FocalLoss()

criterion = nn.L1Loss()
criterion_smooth = nn.SmoothL1Loss(beta=0.1)
criterion_mse = nn.MSELoss()
criterion_class = F.nll_loss
pdist = nn.PairwiseDistance(p=2)
criterion_GAN = nn.MSELoss()

def prediction2label(pred):
	"""Convert ordinal predictions to class labels, e.g.
	
	[0.9, 0.1, 0.1, 0.1] -> 0
	[0.9, 0.9, 0.1, 0.1] -> 1
	[0.9, 0.9, 0.9, 0.1] -> 2
	etc.
	"""
	return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

def ordinal_regression_loss(predictions, targets):
	"""Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

	# Create out modified target with [batch_size, num_labels] shape
	modified_target = torch.zeros_like(predictions)

	# Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
	for i, target in enumerate(targets):
		modified_target[i, 0:target+1] = 1

	return nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1).mean()
#estimate inter-pFC for regularization
def compute_corr_loss(gen_sample,batch_size):
	intra_corr = 0
	count = 0
	for i in range(batch_size):
		start = gen_sample[i, :]
		for j in range(batch_size):
			if (j!=i):
				count +=1
				temp=gen_sample[j,:]

				corr = pearson_correlation(start, temp)
				intra_corr += corr
	return intra_corr/count

def compute_corr_siamese(gen_sample,target,batch_size):
	intra_corr = 0
	count = 0
	for i in range(batch_size):
		start = gen_sample[i, :]
		start = start[torch.triu(torch.ones(start.shape), diagonal=1) == 1] # vectorize upper triangular part of matrix
		start_target = target[i, :]
		start_target = start_target[torch.triu(torch.ones(start_target.shape), diagonal=1) == 1]
		for j in range(batch_size):
			if (j!=i):
				count +=1
				temp=gen_sample[j,:]
				temp = temp[torch.triu(torch.ones(temp.shape), diagonal=1) == 1] # vectorize upper triangular part of matrix
				# corr = correlation_coefficient_loss(start, temp)
				corr = pearson_correlation(start, temp)#1 - correlation_loss(start, temp)
				temp_target=target[j,:]
				temp_target = temp_target[torch.triu(torch.ones(temp_target.shape), diagonal=1) == 1]
				corr_target = pearson_correlation(start_target, temp_target)
				error = torch.abs(corr-corr_target)*(corr-corr_target)
				intra_corr += error
	return intra_corr/count


def r2_loss(output, target):
	target_mean = torch.mean(target)
	ss_tot = torch.sum((target - target_mean) ** 2)
	ss_res = torch.sum((target - output) ** 2)
	r2 = 1 - ss_res / (ss_tot+1e-8)
	return (1-r2) # for loss function

def r2_value(output, target):
	output = np.array(output)
	target = np.array(target)
	target_mean = np.mean(target)
	ss_tot = np.sum((target - target_mean) ** 2)
	ss_res = np.sum((target - output) ** 2)
	r2 = 1 - ss_res / (ss_tot+1e-8)
	return r2

def topk_loss(s,ratio):
	EPS = 1e-10
	if ratio > 0.5:
		ratio = 1-ratio
	s = s.sort(dim=1).values
	res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
	return res

def pearson_correlation(output, target):
	vx = output - torch.mean(output) + 1e-8
	vy = target - torch.mean(target) + 1e-8

	cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) )
	return torch.clamp(cost, min=-1.0, max=1.0)

def correlation_loss(output, target):
	res = pearson_correlation(output, target)
	return (1-res)*(1-res)

def to_categorical(val, num_classes):
	res=np.zeros(num_classes,dtype=int)
	res[val]=1.0
	return res#torch.tensor([val])

def train(model, training_adj, training_labels, training_ids,  training_age, training_scanner, test_adj, test_labels, test_ids, test_age, test_scanner, num_CV, num_rep, lrs, stepsizes, maxval, FC_mean,  args):

	netD = Discriminator_GCN_FC_SC(args)
	
	optimizerG = optim.AdamW(model.parameters(), lr=lrs, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
	optimizerD = optim.AdamW(netD.parameters(), lr=lrs*0.1, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)#, weight_decay=args.weight_decay)

	learning_schedulerG = ReduceLROnPlateau(optimizerG, 'max', factor=0.1, min_lr=1e-6 )
	learning_schedulerD = ReduceLROnPlateau(optimizerD, 'max', factor=0.1, min_lr=1e-6 )

	print_network(model)
	print_network(netD)
	all_epochs_loss = []
	# initialize the early_stopping object
	save_filename = 'train_checkpoint_CV'+ str(num_CV)+ '_rep'+ str(num_rep)+'_lr'+str(lrs)+'_stepsize'+str(stepsizes)+'.pt' 
	save_path = os.path.join(args.save_dir, args.log_name, save_filename)
	if not os.path.exists(os.path.join(args.save_dir, args.log_name)):
		os.makedirs(os.path.join(args.save_dir, args.log_name))
	# torch.save(model.state_dict(), save_path)
	early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.000, path=save_path)



	# check if there's same subject in training/testing
	case_test_ids = []
	case_training_ids = []

	for i in range(len(test_ids)):
		case_test_ids.append(test_ids[i][:-6])
	for i in range(len(training_ids)):
		case_training_ids.append(training_ids[i][:-6])
	delidx = []
	for ii in range(len(case_training_ids)):
		if case_training_ids[ii] in case_test_ids:
			delidx.append(ii)

	if len(delidx)>0:
		print ("Remove %s samples in training set..." %len(delidx))
		for ele in sorted(delidx, reverse = True):
			del training_adj[ele]
			del training_labels[ele]
			del training_ids[ele]
			del training_age[ele]
			del training_scanner[ele]

	# split training/validation set
	idx = np.linspace(0,len(training_ids)-1,len(training_ids),dtype=int)
	train_index, val_index, _, _ = train_test_split(idx, training_age, test_size=0.12, random_state=1)
	# print ("validation index: ", val_index)
	X_train = [training_adj[index] for index in train_index]
	y_train = [training_labels[index] for index in train_index]
	age_train = [training_age[index] for index in train_index]
	scanner_train = [training_scanner[index] for index in train_index]
	ids_train = [training_ids[index] for index in train_index]
	X_val = [training_adj[index] for index in val_index]
	y_val = [training_labels[index] for index in val_index]
	age_val = [training_age[index] for index in val_index]
	scanner_val = [training_scanner[index] for index in val_index]

	mean_FCadj = np.mean(y_train, 0)
	mean_FCadj = torch.from_numpy(mean_FCadj).type(torch.FloatTensor)
	if torch.cuda.is_available():
		mean_FCadj = mean_FCadj.cuda()

	summary_filename = 'CV'+ str(num_CV)+ '_rep'+ str(num_rep)+'_lr'+str(lrs)+'_stepsize'+str(stepsizes)
	writer = SummaryWriter(log_dir=os.path.join(args.save_dir, args.log_name, summary_filename))

	BatchSetSrc1 = np.zeros([args.batch_size, args.lr_dim, args.lr_dim], dtype=np.float32)
	BatchSetTar1 = np.zeros([args.batch_size, args.lr_dim, args.lr_dim], dtype=np.float32)
	BatchSetCls1 = np.zeros([args.batch_size, 4], dtype=np.int32)
	BatchSetScanner1 = np.zeros([args.batch_size, 2], dtype=np.int32)
	BatchSetSrc2 = np.zeros([args.batch_size, args.lr_dim, args.lr_dim], dtype=np.float32)
	BatchSetTar2 = np.zeros([args.batch_size, args.lr_dim, args.lr_dim], dtype=np.float32)
	BatchSetCls2 = np.zeros([args.batch_size, 4], dtype=np.int32)
	BatchSetScanner2 = np.zeros([args.batch_size, 2], dtype=np.int32)
	# compute inter-subject correlation on training set
	reg_train = compute_corr_loss(torch.from_numpy(np.array(y_train)).type(torch.FloatTensor),len(y_train)) ## fixed!! by using FC instead of SC
	print ("inter-subj. corr: ", reg_train)
	FC_mean = torch.from_numpy(np.array(FC_mean)).type(torch.FloatTensor)
	FC_mean = FC_mean.unsqueeze(0).repeat(args.batch_size, 1, 1)
	if torch.cuda.is_available():
		FC_mean = FC_mean.cuda()
	for epoch in range(args.epochs):
		with torch.autograd.set_detect_anomaly(True):

			epoch_loss = []
			epoch_disc_real_loss = []
			epoch_disc_fake_loss = []

			recon_epoch_loss = []
			class_epoch_loss = []
			scanner_epoch_loss = []
			corr_epoch_loss = []
			siamese_epoch_loss = []
			valid_losses = []
			val_cls_losses = []
			val_scanner_losses = []
			val_corr = []
			reg_epoch = []
			bCount = 0

			shuffler = np.random.permutation(len(X_train))
			X_train_shuffled = np.array(X_train)[shuffler]
			y_train_shuffled = np.array(y_train)[shuffler]
			age_train_shuffled = np.array(age_train)[shuffler]
			scanner_train_shuffled = np.array(scanner_train)[shuffler]
			ids_shuffled = np.array(ids_train)[shuffler]
			ids_used = []

			shuffler2 = np.random.permutation(len(X_train))
			X_train_shuffled2 = np.array(X_train)[shuffler2]
			y_train_shuffled2 = np.array(y_train)[shuffler2]
			age_train_shuffled2 = np.array(age_train)[shuffler2]
			scanner_train_shuffled2 = np.array(scanner_train)[shuffler2]
			for p in range(len(X_train)):

				Input_adj1 = X_train_shuffled[p]	
				Output_adj1 = y_train_shuffled[p]	
				gt_class1 = age_train_shuffled[p]
				gt_scanner1 = 0 if scanner_train_shuffled[p]=="A" else 1

				Input_adj2 = X_train_shuffled2[p]	
				Output_adj2 = y_train_shuffled2[p]	
				gt_class2 = age_train_shuffled2[p]
				gt_scanner2 = 0 if scanner_train_shuffled2[p]=="A" else 1


				if np.isnan(gt_class1):
					gt_class1 = 0
	
				if np.isnan(gt_class2):
					gt_class2 = 0
				gt_class1, gt_class2  = int(gt_class1), int(gt_class2)
				gt_class1, gt_class2 = np.array(to_categorical(gt_class1, 4)), np.array(to_categorical(gt_class2, 4))
				gt_scanner1, gt_scanner2 = np.array(to_categorical(gt_scanner1, 2)), np.array(to_categorical(gt_scanner2, 2))

				Input_adj1, Input_adj2 = np.array(Input_adj1), np.array(Input_adj2)
				Output_adj1, Output_adj2 = np.array(Output_adj1), np.array(Output_adj2)
				gt_class1, gt_class2 = np.array(gt_class1), np.array(gt_class2)
				gt_scanner1, gt_scanner2 = np.array(gt_scanner1), np.array(gt_scanner2)
				

				BatchSetSrc1[bCount, :, :] = Input_adj1
				BatchSetTar1[bCount, :, :] = Output_adj1
				BatchSetCls1[bCount] = gt_class1
				BatchSetScanner1[bCount] = gt_scanner1
				BatchSetSrc2[bCount, :, :] = Input_adj2
				BatchSetTar2[bCount, :, :] = Output_adj2
				BatchSetCls2[bCount] = gt_class2
				BatchSetScanner2[bCount] = gt_scanner2
				bCount +=1
				if bCount < args.batch_size-1:
					continue

				Input_adj1 = torch.from_numpy(BatchSetSrc1).type(torch.FloatTensor)
				Output_adj1 = torch.from_numpy(BatchSetTar1).type(torch.FloatTensor)
				gt_class1 = torch.from_numpy(BatchSetCls1).type(torch.LongTensor)
				gt_scanner1 = torch.from_numpy(BatchSetScanner1).type(torch.LongTensor)
				Input_adj2 = torch.from_numpy(BatchSetSrc2).type(torch.FloatTensor)
				Output_adj2 = torch.from_numpy(BatchSetTar2).type(torch.FloatTensor)
				gt_class2 = torch.from_numpy(BatchSetCls2).type(torch.LongTensor)
				gt_scanner2 = torch.from_numpy(BatchSetScanner2).type(torch.LongTensor)
				


				if torch.cuda.is_available():
					Input_adj1 = Input_adj1.cuda()
					Output_adj1 = Output_adj1.cuda()
					gt_class1 = gt_class1.cuda()
					gt_scanner1 = gt_scanner1.cuda()
					Input_adj2 = Input_adj2.cuda()
					Output_adj2 = Output_adj2.cuda()
					gt_class2 = gt_class2.cuda()
					gt_scanner2 = gt_scanner2.cuda()
					
			
				model.train()
				net_AE1, net_class1, net_scanner1 = model(Input_adj1)

				# for siamese training
				net_AE2, net_class2, net_scanner2 = model(Input_adj2)

				
				# -----------------
				#  Train Generator
				# -----------------
				optimizerG.zero_grad()
				recon_loss = (F.mse_loss(net_AE1, Output_adj1) + F.mse_loss(net_AE2, Output_adj2))*0.5


				class_loss = ordinal_regression_loss(net_class1, torch.max(gt_class1, 1)[1]) + ordinal_regression_loss(net_class2, torch.max(gt_class2, 1)[1])
				class_loss *= 0.5
				### focal loss	
				scanner_loss = criterion_focalloss(net_scanner1,  torch.max(gt_scanner1, 1)[1]) + criterion_focalloss(net_scanner2,  torch.max(gt_scanner2, 1)[1])
				scanner_loss *= 0.5

				corr_loss = correlation_loss(net_AE1, Output_adj1) + correlation_loss(net_AE2, Output_adj2)
				corr_loss *= 0.5


				d_real1 = Output_adj1#.detach()
				d_real2 = Output_adj2#.detach()
				gen_loss = criterion_GAN(netD(Input_adj1,net_AE1), torch.ones(args.batch_size, 1).cuda()) + criterion_GAN(netD(Input_adj1,net_AE2), torch.ones(args.batch_size, 1).cuda())
				gen_loss *=0.5
				
				siamese_loss = F.mse_loss(pearson_correlation(Output_adj1, Output_adj2), pearson_correlation(net_AE1, net_AE2))+F.mse_loss(Output_adj1-Output_adj2, net_AE1-net_AE2) #+ args.lam_corr*F.mse_loss(pearson_correlation(Output_adj1, Output_adj2), pearson_correlation(net_AE1, net_AE2))

				lam_adv = args.lam_adv
				generator_loss = recon_loss + lam_adv*gen_loss +args.lam_corr*corr_loss + args.lam_class*class_loss +0.001*scanner_loss +args.lam_siamese * siamese_loss

				reg =  compute_corr_loss(torch.cat([net_AE1, net_AE2],0),args.batch_size*2)
				
				# siamese regularization
				generator_loss += args.lam_reg*(
					F.mse_loss(pearson_correlation(net_AE1, FC_mean),pearson_correlation(Output_adj1, FC_mean))
					+F.mse_loss(pearson_correlation(net_AE2, FC_mean),pearson_correlation(Output_adj2, FC_mean))
					+F.mse_loss((net_AE1-FC_mean),(Output_adj1-FC_mean))
					+F.mse_loss((net_AE2-FC_mean),(Output_adj2-FC_mean))
				)


				generator_loss.backward()
				optimizerG.step()

				# -----------------
				#  Train Discriminator
				# -----------------
				optimizerD.zero_grad()

				real_loss = criterion_GAN(netD(Input_adj1,Output_adj1), torch.ones(args.batch_size,1).cuda()) + criterion_GAN(netD(Input_adj1,Output_adj2), torch.ones(args.batch_size,1).cuda())
				fake_loss = criterion_GAN(netD(Input_adj1,net_AE1.detach()), torch.zeros(args.batch_size, 1).cuda()) + criterion_GAN(netD(Input_adj1,net_AE2.detach()), torch.zeros(args.batch_size, 1).cuda())
				discriminator_loss = 0.5 * (real_loss + fake_loss)
				# if epoch%2 == 0:
				discriminator_loss.backward()
				optimizerD.step()

				



				recon_epoch_loss.append(recon_loss.item())
				class_epoch_loss.append(class_loss.item())
				scanner_epoch_loss.append(scanner_loss.item())
				corr_epoch_loss.append(corr_loss.item())
				siamese_epoch_loss.append(siamese_loss.item())
				epoch_disc_real_loss.append(real_loss.item())
				epoch_disc_fake_loss.append(fake_loss.item())
				epoch_loss.append(generator_loss.item())
				reg_epoch.append(reg.item())
				
				bCount = 0
				BatchSetSrc1.fill(0)
				BatchSetTar1.fill(0)
				BatchSetCls1.fill(0)
				BatchSetScanner1.fill(0)
				BatchSetSrc2.fill(0)
				BatchSetTar2.fill(0)
				BatchSetCls2.fill(0)
				BatchSetScanner2.fill(0)
				ids_used.append(ids_shuffled[p][:-6])
			# validation
			output_AE = []
			input_AE = []

			for p in range(len(X_val)):
				model.eval()
				bCount = 0
				BatchSetSrc1.fill(0)
				BatchSetTar1.fill(0)
				BatchSetCls1.fill(0)
				BatchSetScanner1.fill(0)
				BatchSetSrc2.fill(0)
				BatchSetTar2.fill(0)
				BatchSetCls2.fill(0)
				BatchSetScanner2.fill(0)
				with torch.no_grad():
					Input_adj = X_val[p]	
					Output_adj = y_val[p]	
					gt_class = age_val[p]
					gt_scanner = 0 if scanner_train_shuffled[p]=="A" else 1
					if np.isnan(gt_class):
						gt_class = 0
					gt_class = int(gt_class)
					gt_class = np.array(to_categorical(gt_class, 4))
					gt_scanner = np.array(to_categorical(gt_scanner, 2))
					if epoch==0:
						print (gt_class)
						print (gt_scanner)
					Input_adj = np.array(Input_adj)
					Output_adj = np.array(Output_adj)
					gt_class = np.array(gt_class)
					gt_scanner = np.array(gt_scanner)

					BatchSetSrc1[bCount, :, :] = Input_adj
					BatchSetTar1[bCount, :, :] = Output_adj
					BatchSetCls1[bCount] = gt_class
					BatchSetScanner1[bCount] = gt_scanner
					BatchSetSrc2[bCount, :, :] = Input_adj # same input for evaluation
					BatchSetTar2[bCount, :, :] = Output_adj
					BatchSetCls2[bCount] = gt_class
					BatchSetScanner2[bCount] = gt_scanner

					Input_adj = torch.from_numpy(BatchSetSrc1).type(torch.FloatTensor)
					Output_adj = torch.from_numpy(BatchSetTar1).type(torch.FloatTensor)
					gt_class = torch.from_numpy(BatchSetCls1).type(torch.LongTensor)
					gt_scanner = torch.from_numpy(BatchSetScanner1).type(torch.LongTensor)

					if torch.cuda.is_available():
						Input_adj = Input_adj.cuda()
						Output_adj = Output_adj.cuda()
						gt_class = gt_class.cuda()
						gt_scanner = gt_scanner.cuda()

					net_AE, net_class, net_scanner = model(Input_adj)
					net_AE = net_AE[0]
					Output_adj = Output_adj[0]
					error = criterion_mse(net_AE, Output_adj)

					error_cls = ordinal_regression_loss(net_class[0].unsqueeze(0), torch.max(gt_class[0].unsqueeze(0), 1)[1])
					error_scanner = criterion_class(net_scanner[0].unsqueeze(0), torch.max(gt_scanner[0].unsqueeze(0), 1)[1])

					corr = pearson_correlation(net_AE, Output_adj)
					valid_losses.append(error.item())
					val_cls_losses.append(error_cls.item())
					val_scanner_losses.append(error_scanner.item())
					val_corr.append(corr.item())
					if p == 0:
						output_AE = net_AE.unsqueeze(0)
					else:
						output_AE = torch.cat([output_AE, net_AE.unsqueeze(0)], 0)


				if p==0:
					writer.add_image('Images/val_GT1', Output_adj, epoch, dataformats='HW')
					writer.add_image('Images/val_pred1', net_AE, epoch, dataformats='HW')
				elif p==1:
					writer.add_image('Images/val_GT2', Output_adj, epoch, dataformats='HW')
					writer.add_image('Images/val_pred2', net_AE, epoch, dataformats='HW')
			valid_loss = np.average(valid_losses)
			val_std = np.std(valid_losses)
			valid_cls_loss = np.average(val_cls_losses)
			valid_corr = np.average(val_corr)
			valid_reg = compute_corr_loss(output_AE,len(y_val))

			learning_schedulerG.step(valid_corr) # for ReduceLROnPlateau
			learning_schedulerD.step(valid_corr) # for ReduceLROnPlateau
			if epoch%1==0:
				lr = optimizerG.param_groups[0]['lr']
				
				print("Epoch: ", epoch, 'LR: %.7f' % lr, "recon_loss: %.4f" %np.mean(recon_epoch_loss), "clsloss: %.4f"%np.mean(class_epoch_loss), "scannerloss: %.4f"%np.mean(scanner_epoch_loss), "corrloss: %.4f"%np.mean(corr_epoch_loss), "siamese: %.4f"%np.mean(siamese_epoch_loss), "reg: %.4f"%np.mean(reg_epoch), "Discriminator_real_loss: %.4f" %np.mean(epoch_disc_real_loss), "Discriminator_fake_loss: %.4f" %np.mean(epoch_disc_fake_loss))
				
				print("Epoch: ", epoch, "Val mse: %.4f (+- %.3f)" %(np.mean(valid_losses), np.std(valid_losses)  ), "Val class error: %.3f" %(np.mean(val_cls_losses)), "Val scanner error: %.3f" %(np.mean(val_scanner_losses)), "Val corr: %.4f"%np.mean(val_corr), "Val reg: %.4f"%valid_reg)
			all_epochs_loss.append(np.mean(epoch_loss))

			writer.add_scalar('Loss/train', np.mean(epoch_loss), epoch)
			writer.add_scalar('Loss/Reconstruction', np.mean(recon_epoch_loss), epoch)
			writer.add_scalar('Loss/clsloss', np.mean(class_epoch_loss), epoch)
			writer.add_scalar('Loss/scannerloss', np.mean(scanner_epoch_loss), epoch)
			writer.add_scalar('Loss/corrloss', np.mean(corr_epoch_loss), epoch)
			writer.add_scalar('Loss/siameseloss', np.mean(siamese_epoch_loss), epoch)
			writer.add_scalar('Loss/Discriminator_real', np.mean(epoch_disc_real_loss), epoch)
			writer.add_scalar('Loss/Discriminator_fake', np.mean(epoch_disc_fake_loss), epoch)
			writer.add_scalar('Loss/validation_mse', np.mean(valid_losses), epoch)
			writer.add_scalar('Loss/validation_corr', np.mean(val_corr), epoch)
			writer.add_scalar('Loss/validation_reg', valid_reg, epoch)
			writer.add_scalars('Error',{'train_error':np.mean(epoch_loss),'val_error':np.mean(valid_losses)},  epoch)
			writer.add_scalar('Loss/regularization', np.mean(reg_epoch), epoch)
			
			writer.add_scalar('Learning rate/generator', optimizerG.param_groups[0]['lr'], epoch)
			writer.add_scalar('Learning rate/discriminator', optimizerD.param_groups[0]['lr'], epoch)
			
			torch.save(model.state_dict(), save_path)
			
			writer.close()
	# test:
	# # load the last checkpoint with the best model
	model.load_state_dict(torch.load(save_path))
	# print ("Test at epoch: ", epoch)
	filedir = os.path.join(args.log_path, args.log_name)
	if not os.path.exists(os.path.join(args.log_path, args.log_name)):
		os.makedirs(os.path.join(args.log_path, args.log_name))
	

	output_AE = []
	input_AE = []
	test_losses = []
	test_corr = []
	test_cls_losses = []
	test_scanner_losses = []
	test_SC_arr = []
	test_FC_arr = []
	age_accuracy = 0
	scanner_accuracy = 0
	# for p in range(pLength, len(subjects_adj)):
	for p in range(len(test_ids)):
		model.eval()
		bCount = 0
		BatchSetSrc1.fill(0)
		BatchSetTar1.fill(0)
		BatchSetCls1.fill(0)
		BatchSetScanner1.fill(0)
		with torch.no_grad():
			Input_adj = test_adj[p]	
			Output_adj = test_labels[p]	
			gt_class = test_age[p]
			gt_scanner = 0 if scanner_train_shuffled[p]=="A" else 1
			if np.isnan(gt_class):
				gt_class = 0
			gt_class = int(gt_class)
			gt_class = np.array(to_categorical(gt_class, 4))
			gt_scanner = np.array(to_categorical(gt_scanner, 2))

			test_SC_arr.append(Input_adj)
			test_FC_arr.append(Output_adj)
			Input_adj = np.array(Input_adj)
			Output_adj = np.array(Output_adj)
			gt_class = np.array(gt_class)
			gt_scanner = np.array(gt_scanner)


			BatchSetSrc1[bCount, :, :] = Input_adj
			BatchSetTar1[bCount, :, :] = Output_adj
			BatchSetCls1[bCount] = gt_class
			BatchSetScanner1[bCount] = gt_scanner

			Input_adj = torch.from_numpy(BatchSetSrc1).type(torch.FloatTensor)
			Output_adj = torch.from_numpy(BatchSetTar1).type(torch.FloatTensor)
			gt_class = torch.from_numpy(BatchSetCls1).type(torch.LongTensor)
			gt_scanner = torch.from_numpy(BatchSetScanner1).type(torch.LongTensor)
			
			if torch.cuda.is_available():
				Input_adj = Input_adj.cuda()
				Output_adj = Output_adj.cuda()
				gt_class = gt_class.cuda()
				gt_scanner = gt_scanner.cuda()
				

			net_AE, net_class, net_scanner = model(Input_adj)
			net_AE = net_AE[0]
			Output_adj = Output_adj[0]

			error = criterion_mse(net_AE, Output_adj)
			corr = pearson_correlation(net_AE, Output_adj)
			error_cls = ordinal_regression_loss(net_class[0].unsqueeze(0), torch.max(gt_class[0].unsqueeze(0), 1)[1])
			error_scanner = criterion_class(net_scanner[0].unsqueeze(0), torch.max(gt_scanner[0].unsqueeze(0), 1)[1])

			test_losses.append(error.item())
			test_corr.append(corr.item())
			test_cls_losses.append(error_cls.item())
			test_scanner_losses.append(error_scanner.item())

			
			# renormalize:

			net_AE = net_AE.detach().cpu().float().numpy()*maxval
			net_AE = np.reshape(net_AE, [args.lr_dim, args.lr_dim])
			filename = filedir +'/'+test_ids[p]+'_pred_lr'+str(lrs)+'_stepsize'+str(stepsizes)+'_earlystop.txt'
			with open(filename, "w") as log_file:
				np.savetxt(log_file, net_AE, fmt='%.7f')

			Output_adj = Output_adj.detach().cpu().float().numpy()*maxval
			Output_adj = np.reshape(Output_adj, [args.lr_dim, args.lr_dim])
			
			pred_class = prediction2label(net_class[0].unsqueeze(0).cpu().float().numpy()).squeeze(0)
			print (pred_class, test_age[p])
			# print (pred_class, gt_class[0], test_age[p])
			if pred_class ==test_age[p]:
				age_accuracy +=1

			filename = filedir +'/'+test_ids[p]+'_predcls_lr'+str(lrs)+'_stepsize'+str(stepsizes)+'_earlystop.txt'
			with open(filename, "w") as log_file:
				log_file.write('%s,%s\n' % (pred_class, test_age[p]))

			# for scanner	
			_, pred_class = torch.max(net_scanner[0].unsqueeze(0), 1)
			pred_class = (pred_class.squeeze(0)).cpu().float().numpy()
			pred_class = int(pred_class)
			gt_scanner = 0 if test_scanner[p]=="A" else 1
			print (pred_class, gt_scanner)
			if pred_class ==gt_scanner:
				scanner_accuracy +=1

			filename = filedir +'/'+test_ids[p]+'_predscanner_lr'+str(lrs)+'_stepsize'+str(stepsizes)+'_earlystop.txt'
			with open(filename, "w") as log_file:
				log_file.write('%s,%s\n' % (pred_class, gt_scanner))

	print("Test recon error: %.4f (+-%.3f)" %(np.mean(test_losses), np.std(test_losses)))
	print("Test correlation: %.4f (+-%.3f)" %(np.mean(test_corr), np.std(test_corr)))

	print("Test age accuracy: %.3f " %(age_accuracy/len(test_ids)))
	print("Test scanner accuracy: %.3f " %(scanner_accuracy/len(test_ids)))
	
	return np.mean(test_losses)
