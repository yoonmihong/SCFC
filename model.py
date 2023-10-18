import torch
import torch.nn as nn
# from layers import *
from ops import *

import torch.nn.functional as F



class SC_FC_age_ordinal_scanner_GraphConv_siamese(nn.Module):

	def __init__(self, args):
		super(SC_FC_age_ordinal_scanner_GraphConv_siamese, self).__init__()

		self.lr_dim = args.lr_dim
		self.hr_dim = args.hr_dim
		self.batch_size = args.batch_size

		self.net = ChebGraphConv_SC_FC_age_ordinal_scanner(self.lr_dim, self.batch_size)

		
		if torch.cuda.is_available():
			self.net.cuda()

	def batch_eye(self):
		batch_size = self.batch_size
		n =self.lr_dim
		I = torch.eye(n).unsqueeze(0)
		I = I.repeat(batch_size, 1, 1)
		return I


	def forward(self, A, is_training=True):
		with torch.autograd.set_detect_anomaly(True):

			I =  self.batch_eye()

			A = A.type(torch.FloatTensor)

			if torch.cuda.is_available():
				I = I.cuda()
				A = A.cuda()  
				
			self.network_outs1, self.age_outs1, self.scanner_outs1 = self.net(A, I)

		return self.network_outs1, self.age_outs1, self.scanner_outs1



class Discriminator_GCN_FC_SC(nn.Module):
	def __init__(self, args):
		super(Discriminator_GCN_FC_SC, self).__init__()
		self.batch_size = args.batch_size
		self.lr_dim = args.lr_dim
		self.hr_dim = args.hr_dim
		self.net = GCN_Discriminator(self.lr_dim, self.batch_size)
		if torch.cuda.is_available():
			self.net.cuda()

	def batch_eye(self):
		batch_size = self.batch_size
		n =self.lr_dim
		I = torch.eye(n).unsqueeze(0)
		I = I.repeat(batch_size, 1, 1).to(I.device).float() 
		return I
	def forward(self, A, inputs):

		if torch.cuda.is_available():
			inputs = inputs.cuda()
			A = A.cuda()   
		output = self.net(A, inputs)
		# output = self.sigmoid(output)
		return output#torch.abs(output)



def print_network(net):
	num_params = 0
	# for param in net.parameters():
	for name, param in net.state_dict().items():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)