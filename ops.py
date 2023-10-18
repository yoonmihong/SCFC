#
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# from layers import *
import scipy.sparse as sp

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math

class GraphConvolution(Module):

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.matmul(input, self.weight)
		output = torch.einsum('bij,bjd->bid', [adj, support])
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'

class WeightedSum(nn.Module):
	def __init__(self):
		super(WeightedSum, self).__init__()
		self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
		self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
		self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
		nn.init.normal_(self.w1, mean=0.0, std=0.1)
		nn.init.normal_(self.w2, mean=0.0, std=0.1)
		nn.init.normal_(self.w3, mean=0.0, std=0.1)
	def forward(self, x1, x2, x3):
		out = self.w1*x1 + self.w2*x2 + self.w3*x3
		return out, self.w1, self.w2, self.w3

class GraphUnpool(nn.Module):

	def __init__(self):
		super(GraphUnpool, self).__init__()

	def forward(self, A, X, idx):
		new_X = torch.zeros([A.shape[0], X.shape[1]]).cuda()
		new_X[idx] = X
		return A, new_X


class GraphPool(nn.Module):

	def __init__(self, k, in_dim):
		super(GraphPool, self).__init__()
		self.k = k
		self.proj = nn.Linear(in_dim, 1, bias=False)
		self.sigmoid = nn.Sigmoid()
		self.in_dim = in_dim
		self.reset_parameters()

	def reset_parameters(self):
		bound = 1.0 / math.sqrt(self.in_dim)
		if isinstance(self, nn.Linear):
			# torch.nn.init.xavier_uniform_(self.weight)
			torch.nn.init.uniform_(self.weight, -bound, bound )


	def forward(self, A, X):
		scores = self.proj(X)
		weight = self.proj.weight
		scores /= weight.norm(p=2, dim=-1)
		# scores = torch.abs(scores)
		scores = torch.squeeze(scores)
		# print ("Mean scores: ", torch.mean(scores))
		# scores = (scores-torch.mean(scores))/torch.std(scores)
		scores = self.sigmoid(scores)
		# scores = self.sigmoid(scores/100)
		num_nodes = A.shape[0]
		values, idx = torch.topk(scores, int(self.k*num_nodes))
		new_X = X[idx, :]
		values = torch.unsqueeze(values, -1)
		new_X = torch.mul(new_X, values)
		A = A[idx, :]
		A = A[:, idx]
		return A, new_X, idx, scores, weight

### from g-U-net
class Pool(nn.Module):

	def __init__(self, k, in_dim, p=0):
		super(Pool, self).__init__()
		self.k = k
		self.sigmoid = nn.Sigmoid()
		self.proj = nn.Linear(in_dim, 1)
		self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

	def forward(self, g, h):
		Z = self.drop(h)
		weights = self.proj(Z).squeeze()
		scores = self.sigmoid(weights/100) # need to check
		return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

	def __init__(self, *args):
		super(Unpool, self).__init__()

	def forward(self, g, h, pre_h, idx):
		new_h = h.new_zeros([g.shape[0], h.shape[1]])
		new_h[idx] = h
		
		return g, new_h


def top_k_graph(scores, g, h, k):
	num_nodes = g.shape[0]
	values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
	new_h = h[idx, :]
	values = torch.unsqueeze(values, -1)
	new_h = torch.mul(new_h, values)
	un_g = g.bool().float()
	un_g = torch.matmul(un_g, un_g).bool().float()
	un_g = un_g[idx, :]
	un_g = un_g[:, idx]
	g = norm_g(un_g)
	return g, new_h, idx


def norm_g(g):
	degrees = torch.sum(g, 1)
	g = g / degrees
	return g

class ChebGraphConv(nn.Module):
	def __init__(self, K, in_features, out_features, bias=True):
		super(ChebGraphConv, self).__init__()
		self.K = K
		self.weight = nn.Parameter(torch.FloatTensor(K+1, in_features, out_features))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.in_features = in_features
		self.out_features = out_features
		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		# torch.nn.init.xavier_uniform(self.weight)
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
			torch.nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x, gso):
		# Chebyshev polynomials:
		# x_0 = x,
		# x_1 = gso * x,
		# x_k = 2 * gso * x_{k-1} - x_{k-2},
		# where gso = 2 * gso / eigv_max - id.

		cheb_poly_feat = []
		if self.K < 0:
			raise ValueError('ERROR: The order of Chebyshev polynomials shoule be non-negative!')
		elif self.K == 0:
			# x_0 = x
			cheb_poly_feat.append(x)
		elif self.K == 1:
			# x_0 = x
			cheb_poly_feat.append(x)
			if gso.is_sparse:
				# x_1 = gso * x
				cheb_poly_feat.append(torch.sparse.mm(gso, x))
			else:
				if x.is_sparse:
					x = x.to_dense
				# x_1 = gso * x
				# cheb_poly_feat.append(torch.mm(gso, x))
				cheb_poly_feat.append(torch.matmul(gso, x))
		else:
			# x_0 = x
			cheb_poly_feat.append(x)
			if gso.is_sparse:
				# x_1 = gso * x
				cheb_poly_feat.append(torch.sparse.mm(gso, x))
				# x_k = 2 * gso * x_{k-1} - x_{k-2}
				for k in range(2, self.K+1):
					cheb_poly_feat.append(torch.sparse.mm(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
			else:
				if x.is_sparse:
					x = x.to_dense
				# x_1 = gso * x
				cheb_poly_feat.append(torch.matmul(gso, x)) ## may have different results (not deterministic)
				# x_k = 2 * gso * x_{k-1} - x_{k-2}
				for k in range(2, self.K+1):
					cheb_poly_feat.append(torch.matmul(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
		# print (cheb_poly_feat.shape)
		# feature = torch.stack(cheb_poly_feat, dim=0)
		feature = torch.stack(cheb_poly_feat, dim=1)
		# print (feature.shape)
		if feature.is_sparse:
			feature = feature.to_dense()
		# cheb_graph_conv = torch.einsum('bij,bjk->ik', feature, self.weight)
		cheb_graph_conv = torch.einsum('bdij,djk->bik', feature, self.weight)

		if self.bias is not None:
			# cheb_graph_conv = torch.add(input=cheb_graph_conv, other=self.bias, alpha=1)
			cheb_graph_conv = cheb_graph_conv + self.bias
		else:
			cheb_graph_conv = cheb_graph_conv

		return cheb_graph_conv

	def extra_repr(self) -> str:
		return 'K={}, in_features={}, out_features={}, bias={}'.format(
			self.K, self.in_features, self.out_features, self.bias is not None
		)


###
class GCN(nn.Module):

	def __init__(self, in_dim, out_dim, dropout=0, act=F.relu, norm=None):
		super(GCN, self).__init__()
		self.proj = nn.Linear(in_dim, out_dim)
		self.drop = nn.Dropout(p=dropout)
		self.act = act
		# self.norm = nn.InstanceNorm1d(out_dim) if norm=='instance' else None
		if norm == 'instance':
			self.norm = nn.InstanceNorm1d(out_dim, affine=True)
		elif norm == 'layer':
			self.norm = LayerNorm(out_dim)
		else:
			self.norm = None

	def Normalize(self, X):
		X = torch.transpose(X, 0, 1) # CxN
		# print (X.shape)
		X = X.unsqueeze(0) # 1xCxN
		X = self.norm(X)
		X = X.squeeze(0)
		X = torch.transpose(X, 0, 1) # NxC
		# print (X.shape)
		return X

	def forward(self, A, X):

		X = self.drop(X)
		X = torch.matmul(A, X)
		X = self.proj(X)
		if self.norm:
			X = self.Normalize(X.contiguous())
		X = self.act(X)
		return X



class LayerNorm(nn.Module):
	def __init__(self, num_features, eps=1e-05, elementwise_affine=True):
		super(LayerNorm, self).__init__()

		self.LayerNorm = nn.LayerNorm(num_features, eps, elementwise_affine)#num_features = (input.size()[1:])

	def forward(self, x):
		x = self.LayerNorm(x)
		return x

class BatchNorm(nn.Module):
	def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
		super(BatchNorm, self).__init__()

		self.batchnorm_layer = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

	def forward(self, x):
		x = x.permute(0, 2, 1)
		x = self.batchnorm_layer(x)
		x = x.permute(0, 2, 1)
		return x

class GraphUnet(nn.Module):

	def __init__(self, ks, in_dim, out_dim, dim=320):
		super(GraphUnet, self).__init__()
		self.ks = ks

		self.start_gcn = GCN(in_dim, dim)
		self.bottom_gcn = GCN(dim, dim)
		self.end_gcn = GCN(2*dim, out_dim)
		self.down_gcns = []
		self.up_gcns = []
		self.pools = []
		self.unpools = []
		self.l_n = len(ks)
		for i in range(self.l_n):
			self.down_gcns.append(GCN(dim, dim))
			self.up_gcns.append(GCN(dim, dim))
			self.pools.append(GraphPool(ks[i], dim))
			self.unpools.append(GraphUnpool())

	def forward(self, A, X):
		adj_ms = []
		indices_list = []
		down_outs = []
		X = self.start_gcn(A, X)
		start_gcn_outs = X
		org_X = X
		for i in range(self.l_n):

			X = self.down_gcns[i](A, X)
			adj_ms.append(A)
			down_outs.append(X)
			A, X, idx = self.pools[i](A, X)
			indices_list.append(idx)
		X = self.bottom_gcn(A, X)
		for i in range(self.l_n):
			up_idx = self.l_n - i - 1

			A, idx = adj_ms[up_idx], indices_list[up_idx]
			A, X = self.unpools[i](A, X, idx)
			X = self.up_gcns[i](A, X)
			X = X.add(down_outs[up_idx])
		X = torch.cat([X, org_X], 1)
		X = self.end_gcn(A, X)

		return X, start_gcn_outs

class GCN_Discriminator(nn.Module):

	def __init__(self, in_dim, batch_size, dim=64):
		super(GCN_Discriminator, self).__init__()
		self.batch = batch_size
		hidden_dim = 8
		self.start_gcn = ChebGraphConv(1, in_dim, dim)
		self.bottom_gcn = ChebGraphConv(1, dim, 1)

		self.proj1 = nn.Linear(int(in_dim*1), 64)
		self.proj2 = nn.Linear(64, 1) 

		self.out = nn.Tanh() 
		self.drop = nn.Dropout(p=0.1)
		self.lrelu = nn.LeakyReLU(0.3)
		
	def normalize_adj_torch(self, mx):
		rowsum = mx.sum(1)
		r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
		r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
		r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
		mx = torch.matmul(mx, r_mat_inv_sqrt)
		mx = torch.transpose(mx, 0, 1)
		mx = torch.matmul(mx, r_mat_inv_sqrt)
		return mx
	
	def normalize_adj(self, A):
		for b in range(self.batch):
			tmp = A[b,:,:].clone() # to avoid runtime error

			tmp = normalize_adj_torch(tmp)
			tmp = tmp + torch.eye(tmp.shape[0]).to(tmp.device).float() 
			tmp = normalize_adj_torch(tmp)

			A[b,:,:] = tmp
		return A

	def forward(self, A, X):
		
		X = self.start_gcn(X, A)
		X = nn.LeakyReLU(0.2, True)(X)
		X = self.bottom_gcn(X,A)
		X = nn.LeakyReLU(0.2, True)(X)
		
		
		X = X.contiguous().view(self.batch, -1)
		X = self.proj1(X)
		X = nn.LeakyReLU(0.2, True)(X)

		X = self.proj2(X)
		outputs = X
		return outputs



class ChebGraphConv_SC_FC_age_ordinal_scanner(nn.Module):

	def __init__(self, in_dim, batch_size, dim=64):
		super(ChebGraphConv_SC_FC_age_ordinal_scanner, self).__init__()
		self.in_dim = in_dim
		self.batch = batch_size
		dim1 = 64
		dim2 = 64
		dim3 = 64
		self.start_gcn = ChebGraphConv(5, in_dim, dim1)
		self.bottom_gcn = ChebGraphConv(5, dim1, dim2)
		self.end_gcn = ChebGraphConv(5, dim2, int(dim3))

		self.norm1 = LayerNorm([in_dim, dim1])
		self.norm2= LayerNorm([in_dim, dim2])
		self.proj1 = nn.Linear(int((dim1+dim2)*2), int(dim))
		self.proj2 = nn.Linear(int(dim), int(dim)) 
		self.proj3 = nn.Linear(int(dim), 4) # age classification
		self.proj1_scan = nn.Linear(int((dim1+dim2)*2), int(dim))
		self.proj2_scan = nn.Linear(int(dim), int(dim))
		self.proj4 = nn.Linear(int(dim), 2) # scanner classification

		self.out = nn.Sigmoid()
		self.drop = nn.Dropout(p=0.1) # automatically turn off for evaluation mode
		self.lrelu = nn.LeakyReLU(0.2)
		self.relu = nn.ReLU(inplace=False)

	def batch_eye(self):
		batch_size = self.batch
		n =self.in_dim
		I = torch.eye(n).unsqueeze(0)
		I = I.repeat(batch_size, 1, 1)
		return I
	
	def normalize_output(self, out):
		for b in range(self.batch):
			tmp = out[b,:,:]
			tmp = tmp.fill_diagonal_(0)
			tmp = (tmp + tmp.t())/2
			out[b,:,:] = tmp
		return out
	def forward(self, A, X):

		X = self.start_gcn(X, A)
		X = self.norm1(X)
		# X = self.drop(X)
		X = self.relu(X)
		X0 = X.view(self.batch, self.in_dim, -1)
		Z1 = torch.mean(X0, dim=1, keepdim=True)
		Z2,_ = torch.max(X0, dim=1, keepdim=True)
		Z = torch.cat([Z1, Z2],2)

		X = self.bottom_gcn(X,A)
		X = self.norm2(X)
		# X = self.drop(X)
		X = self.relu(X)
		X0 = X.view(self.batch, self.in_dim, -1)
		Z1 = torch.mean(X0, dim=1, keepdim=True)
		Z2,_ = torch.max(X0, dim=1, keepdim=True)
		Z = torch.cat([Z, Z1, Z2],2)

		# AE_out = F.relu(self.end_gcn(X, A))
		AE_out = (self.end_gcn(X, A))
		X0 = AE_out
		AE_out = (torch.einsum('bij,bjk->bik', AE_out, torch.transpose(AE_out, 1, 2)))
		AE_out = self.relu(AE_out).clone() # for pytorch1.12.0 version
		AE_out = self.normalize_output(AE_out)

		Z = Z.view(self.batch,-1)
		latent = Z
		Z_scan = Z
		Z = self.proj1(Z)

		Z = self.lrelu(Z)
		Z = self.proj2(Z)
		Z = self.lrelu(Z)
		out_class = self.proj3(Z)
		out_class = self.out(out_class) # for ordinal regression, use sigmoid for each node

		Z_scan = self.proj1_scan(Z_scan)

		Z_scan = self.lrelu(Z_scan)
		Z_scan = self.proj2_scan(Z_scan)
		Z_scan = self.lrelu(Z_scan)
		scanner_class = self.proj4(Z_scan)
		scanner_class = F.log_softmax(scanner_class,dim=1)
		return AE_out, out_class, scanner_class


### final prediction model for journal paper
class ChebGraphConv_SC_FC_age_ordinal_scanner_siamese(nn.Module):

	def __init__(self, in_dim, batch_size, dim=64):
		super(ChebGraphConv_SC_FC_age_ordinal_scanner_siamese, self).__init__()
		self.in_dim = in_dim
		self.batch = batch_size
		hidden_dim = 8
		dim1 = 64
		dim2 = 64
		dim3 = 64
		self.start_gcn = ChebGraphConv(1, in_dim, dim1)#RaGCN_batch(in_dim, in_dim, hidden_dim, dim, self.batch)#ChebGraphConv(5, in_dim, dim)
		self.bottom_gcn = ChebGraphConv(1, dim1, dim2)#RaGCN_batch(in_dim, dim, hidden_dim, int(dim), self.batch)#GraphConvolution(dim*2, dim)
		# self.bottom_gcn2 = ChebGraphConv(5, dim, dim)#RaGCN_batch(in_dim, dim, hidden_dim, int(dim), self.batch)
		# self.bottom_gcn3 = ChebGraphConv(5, dim, dim)#RaGCN_batch(in_dim, dim, hidden_dim, int(dim), self.batch)
		# self.bottom_gcn4 = ChebGraphConv(5, dim, dim)#RaGCN_batch(in_dim, dim, hidden_dim, int(dim), self.batch)
		self.end_gcn = ChebGraphConv(1, dim2, int(dim3))#RaGCN_batch(in_dim, int(dim), hidden_dim, int(dim), self.batch)#GraphConvolution(dim, in_dim)
		# self.start_gcn = GraphConvolution(in_dim, dim)
		# self.bottom_gcn = GraphConvolution(dim, dim)
		# self.end_gcn = GraphConvolution(dim, dim)
		# self.end_gcn_ge = RaGCN(in_dim, dim, hidden_dim, 1)
		# self.norm1 = LayerNorm([in_dim, dim1])#BatchNorm(dim)#LayerNorm([in_dim, dim])
		# self.norm2= LayerNorm([in_dim, dim2])#BatchNorm(dim)#LayerNorm([in_dim, dim])
		# self.proj1 = nn.Linear(int(in_dim/batch_size), int(dim*2))
		self.proj1 = nn.Linear(int((dim1+dim2)*2), int(dim))
		self.proj2 = nn.Linear(int(dim), int(dim)) 
		self.proj3 = nn.Linear(int(dim), 4) # age classification
		self.proj1_scan = nn.Linear(int((dim1+dim2)*2), int(dim))
		self.proj2_scan = nn.Linear(int(dim), int(dim))
		self.proj4 = nn.Linear(int(dim), 2) # scanner classification

		self.out = nn.Sigmoid()
		self.drop = nn.Dropout(p=0.1) # automatically turn off for evaluation mode
		self.lrelu = nn.LeakyReLU(0.2)
		self.relu = nn.ReLU(inplace=False)
		# self.num_nodes = int(in_dim/batch_size)

	def batch_eye(self):
		batch_size = self.batch
		n =self.in_dim
		I = torch.eye(n).unsqueeze(0)
		I = I.repeat(batch_size, 1, 1)
		return I
	# def normalize_adj(self, A):
	# 	for b in range(self.batch):
	# 		tmp = A[b,:,:]
	# 		tmp = tmp.fill_diagonal_(0)
	# 		# # SC row-wise normalization
	# 		tmp = normalize_adj_torch(tmp)
	# 		# # min-max normalization
	# 		# print (tmp.shape)
	# 		# minval,_ = torch.min(tmp)
	# 		# maxval,_ = torch.max(tmp)
	# 		# tmp = (tmp-minval)/(maxval -minval)
	# 		# Kipf: renomalization trick
	# 		tmp = tmp + torch.eye(tmp.shape[0]).to(tmp.device).float() 
	# 		tmp = normalize_adj_torch(tmp)
	# 		# tmp = tmp.fill_diagonal_(1)
	# 		# tmp = (tmp + tmp.t())/2
	# 		A[b,:,:] = tmp
	# 	return A
	# def calc_chebynet_gso(self, A):
	# 	for b in range(self.batch):
	# 		tmp = A[b,:,:]
	# 		tmp = torch.eye(tmp.shape[0]).to(tmp.device).float() - tmp
	# 		eigval_max = torch.max(eigvals(a=tmp).real)
	# 		# eigval_max = torch.max(torch.linalg.eigvals(tmp).real)
	# 		tmp = 2*tmp/eigval_max - torch.eye(tmp.shape[0]).to(tmp.device).float()
	# 		A[b,:,:] = tmp
	# 	return A
	def normalize_output(self, out):
		for b in range(self.batch):
			tmp = out[b,:,:]
			tmp = tmp.fill_diagonal_(0)
			tmp = (tmp + tmp.t())/2
			out[b,:,:] = tmp
		return out
	def forward_once(self, A, X):
		# I = torch.eye(X.shape[1]).unsqueeze(0).to(X.device).float() 
		# I = I.repeat(X.shape[0], 1, 1)
		# X = A+I
		# A = self.drop(A)

		# A = A.fill_diagonal_(0)
		# # A = F.normalize(A, p=2, dim=1)
		# A = normalize_adj_torch(A)
		# A = A.fill_diagonal_(1)#.0+1e-10)
		# A = (A + A.t())/2
		# A = self.normalize_adj(A)
		# A = self.calc_chebynet_gso(A)

		X = self.start_gcn(X, A)
		# X = self.norm1(X)
		# X = self.drop(X)
		X = self.relu(X)
		X0 = X.view(self.batch, self.in_dim, -1)
		Z1 = torch.mean(X0, dim=1, keepdim=True)
		Z2,_ = torch.max(X0, dim=1, keepdim=True)
		Z = torch.cat([Z1, Z2],2)

		X = self.bottom_gcn(X,A)
		# X = self.norm2(X)
		# X = self.drop(X)
		X = self.relu(X)
		X0 = X.view(self.batch, self.in_dim, -1)
		Z1 = torch.mean(X0, dim=1, keepdim=True)
		Z2,_ = torch.max(X0, dim=1, keepdim=True)
		Z = torch.cat([Z, Z1, Z2],2)

		# X = self.bottom_gcn2(X,A)
		# # X = self.drop(X)
		# X = F.relu(X)
		# X0 = X.view(self.batch, self.in_dim, -1)
		# Z1 = torch.mean(X0, dim=1, keepdim=True)
		# Z2,_ = torch.max(X0, dim=1, keepdim=True)
		# Z = torch.cat([Z, Z1, Z2],2)
		# X = self.bottom_gcn3(X,A)
		# # X = self.drop(X)
		# X = F.relu(X)
		# X0 = X.view(self.batch, self.in_dim, -1)
		# Z1 = torch.mean(X0, dim=1, keepdim=True)
		# Z2,_ = torch.max(X0, dim=1, keepdim=True)
		# Z = torch.cat([Z, Z1, Z2],2)
		# X = self.bottom_gcn4(X,A)
		# # X = self.drop(X)
		# X = F.relu(X)
		# X0 = X.view(self.batch, self.in_dim, -1)
		# Z1 = torch.mean(X0, dim=1, keepdim=True)
		# Z2,_ = torch.max(X0, dim=1, keepdim=True)
		# Z = torch.cat([Z, Z1, Z2],2)

		# AE_out = F.relu(self.end_gcn(X, A))
		AE_out = (self.end_gcn(X, A))
		X0 = AE_out
		AE_out = (torch.einsum('bij,bjk->bik', AE_out, torch.transpose(AE_out, 1, 2)))
		AE_out = self.relu(AE_out).clone() # for pytorch1.12.0 version
		AE_out = self.normalize_output(AE_out)

		# Evc_out = F.relu(self.end_gcn(X, A)[:,-1])
		# adj = torch.where(A > 0.0001, A, torch.zeros_like(A))
		# X = self.end_gcn_ge(AE_out,adj)
		# X = X.view(self.batch, self.in_dim, -1)
		# Z = torch.mean(X, dim=1, keepdim=True)
		# Z1,_ = torch.max(X, dim=1, keepdim=True)
		# Z = torch.cat([Z, Z1],2)
		# print (X.shape)
		# Z = torch.mean(X, dim=1, keepdim=True) # global average pooling
		# Z1,_ = torch.max(X, dim=1, keepdim=True) # global max pooling
		# Z = torch.cat([Z, Z1],2)
		# print (Z.shape)
		# Z = torch.reshape(Z, (self.batch,self.in_dim))
		# Z = torch.mean(Z, dim=1, keepdim=True)
		# ge = F.relu(Z)

		# X0 = X0.view(self.batch, self.in_dim, -1)
		# Z1 = torch.mean(X0, dim=1, keepdim=True)
		# Z2,_ = torch.max(X0, dim=1, keepdim=True)
		# Z = torch.cat([Z, Z1, Z2],2)

		Z = Z.view(self.batch,-1)
		latent = Z
		Z_scan = Z
		Z = self.proj1(Z)

		Z = self.lrelu(Z)
		Z = self.proj2(Z)
		Z = self.lrelu(Z)
		out_class = self.proj3(Z)
		out_class = self.out(out_class) # for ordinal regression, use sigmoid for each node

		Z_scan = self.proj1_scan(Z_scan)

		Z_scan = self.lrelu(Z_scan)
		Z_scan = self.proj2_scan(Z_scan)
		Z_scan = self.lrelu(Z_scan)
		scanner_class = self.proj4(Z_scan)
		scanner_class = F.log_softmax(scanner_class,dim=1)
		return AE_out, out_class, scanner_class

	def forward(self, A1, X1, A2, X2):
		AE_out1, out_class1, scanner_class1 = self.forward_once(A1, X1)
		AE_out2, out_class2, scanner_class2 = self.forward_once(A2, X2)
		return AE_out1, out_class1, scanner_class1, AE_out2, out_class2, scanner_class2


