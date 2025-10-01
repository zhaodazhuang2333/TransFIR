import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv
from torch_geometric.utils import degree
from torch_geometric.nn.conv import MessagePassing
import pdb
import math
from torch_geometric.utils import scatter
from sklearn.cluster import KMeans

class EarlyStopping:
	def __init__(self, patience=10, min_delta=0, path="checkpoint.pth", verbose=False):
		self.patience = patience
		self.min_delta = min_delta
		self.path = path
		self.verbose = verbose
		self.counter = 0
		self.best_loss = None
		self.early_stop = False
		self.best_model_state = None

	def __call__(self, val_loss, model):
		if self.best_loss is None:
			self.best_loss = val_loss
			self.save_checkpoint(model)
		elif val_loss < self.best_loss - self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
			self.save_checkpoint(model)
		else:
			self.counter += 1
			if self.verbose:
				print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
			if self.counter >= self.patience:
				self.early_stop = True

	def save_checkpoint(self, model):
		self.best_model_state = model.state_dict()
		torch.save(self.best_model_state, self.path)
		if self.verbose:
			print(f"Validation loss decreased. Saving model to {self.path}")

	def load_checkpoint(self, model):
		model.load_state_dict(torch.load(self.path))


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=512):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # (1, max_len, d_model)
		self.register_buffer('pe', pe)

	def forward(self, x):
		return x + self.pe[:, :x.size(1)]
	

class TransformerModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers):
		super(TransformerModel, self).__init__()
		# Transformer layers
		self.transformer_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first = True)
		self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers=num_layers)
		self.output_layer = nn.Linear(input_dim, output_dim)
		self.positional_encoding = PositionalEncoding(input_dim)

	def forward(self, x):
		# Transformer forward passx = self.pos_encoder(x)
		x = self.positional_encoding(x)
		x = self.transformer(x)
		# output = x.mean(dim=1)  # Global average pooling
		return x
	



class TransformerEncoderLayer_with_query(nn.Module):
	def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
		super(TransformerEncoderLayer_with_query, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
		self.linear1 = nn.Linear(d_model, d_ff)
		self.linear2 = nn.Linear(d_ff, d_model)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.norm_for_query = nn.LayerNorm(d_model)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.positional_encoding = PositionalEncoding(d_model)

	def forward(self, src, query=None):
		# self-attention block

		src2 = self.norm1(src)
		src2 = self.self_attn(src2, src2, src2)[0]
		src = src + self.dropout(src2)
		if query is not None:
			query = query.unsqueeze(1)
			query_norm = self.norm_for_query(query)
			src_norm = self.norm2(src)
			cross_output = self.cross_attn(query_norm, src_norm, src_norm)[0]
			src = src + self.dropout(cross_output)

		# feedforward block
		src2 = self.norm3(src)
		ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
		src = src + self.dropout(ff_output)
		return src


class TransformerEncoder_with_query(nn.Module):
	def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
		super(TransformerEncoder_with_query, self).__init__()
		self.layers = nn.ModuleList([TransformerEncoderLayer_with_query(d_model, num_heads, d_ff, dropout)
									 for _ in range(num_layers)])
		# self.positional_encoding = PositionalEncoding(d_model)
	def forward(self, src, query=None):
		# src = self.positional_encoding(src)
		for layer in self.layers:
			src = layer(src, query)
		return src
	

def sort_by_last_dim_with_neg1_last(arr):
	# arr: shape (B, N, 4)
	last_val = arr[:, :, 3]  # shape (B, N)

	# 构造排序键：将 -1 替换为 +inf（放到最后）
	sort_key = torch.where(last_val == -1, torch.tensor(float('inf'), device=arr.device), last_val)

	# 获取排序索引
	sorted_indices = torch.argsort(sort_key, dim=1)

	# 扩展 sorted_indices 用于 index_select
	B, N, _ = arr.shape
	batch_indices = torch.arange(B, device=arr.device).unsqueeze(1).expand(B, N)

	# 对 N 维度排序
	sorted_arr = arr[batch_indices, sorted_indices]

	return sorted_arr




def get_embedding_two_hops(one_hop_chain, two_hop_chain, entity_embedding, relation_embedding, empty_embedding, time_projection, device='cuda'):
	# one_hop_chain = torch.cat(one_hop_chain,)
	pdb.set_trace()
	one_hop_chain = one_hop_chain.to(torch.int64)
	two_hop_chain = two_hop_chain.to(torch.int64)
	one_hop_mask = one_hop_chain[:, :, 0] == -1
	two_hop_mask = (two_hop_chain[:, :, 0] == -1) & (two_hop_chain[:, :, 4] == -1)
	one_hop_chain = one_hop_chain[~one_hop_mask]
	two_hop_chain = two_hop_chain[~two_hop_mask]
	one_hop_chain_embedding = torch.zeros(len(one_hop_chain), 8, entity_embedding.shape[1], device=entity_embedding.device)
	two_hop_chain_embedding = torch.zeros(len(two_hop_chain), 8, entity_embedding.shape[1], device=entity_embedding.device)
	one_hop_chain_embedding[:, 0] = entity_embedding[one_hop_chain[:, 0]]
	one_hop_chain_embedding[:, 1] = relation_embedding[one_hop_chain[:, 1]]
	one_hop_chain_embedding[:, 2] = entity_embedding[one_hop_chain[:, 2]]
	one_hop_chain_embedding[:, 3] = time_projection(one_hop_chain[:, 3].unsqueeze(-1).float())

	two_hop_chain_embedding[:, 0] = entity_embedding[two_hop_chain[:, 0]]
	two_hop_chain_embedding[:, 1] = relation_embedding[two_hop_chain[:, 1]]
	two_hop_chain_embedding[:, 2] = entity_embedding[two_hop_chain[:, 2]]
	two_hop_chain_embedding[:, 3] = time_projection(two_hop_chain[:, 3].unsqueeze(-1).float())
	two_hop_chain_embedding[:, 4] = entity_embedding[two_hop_chain[:, 4]]
	two_hop_chain_embedding[:, 5] = relation_embedding[two_hop_chain[:, 5]]
	two_hop_chain_embedding[:, 6] = entity_embedding[two_hop_chain[:, 6]]
	two_hop_chain_embedding[:, 7] = time_projection(two_hop_chain[:, 7].unsqueeze(-1).float())
	chain_embedding = torch.cat((one_hop_chain_embedding, two_hop_chain_embedding), dim=0)
	mask = torch.cat((one_hop_mask, two_hop_mask), dim=1)

	chain_embedding = one_hop_chain_embedding
	mask = one_hop_mask
	return chain_embedding, mask

# def get_embedding_only_one_hop(one_hop_chain, entity_embedding, relation_embedding, empty_embedding, cls_embedding, time_projection, device='cuda'):
#     # one_hop_chain = torch.cat(one_hop_chain,)
	
#     B, N, M = one_hop_chain.shape[0], one_hop_chain.shape[1], one_hop_chain.shape[2]
#     one_hop_chain = one_hop_chain.to(torch.int64)
#     one_hop_chain = sort_by_last_dim_with_neg1_last(one_hop_chain)

#     one_hop_mask = one_hop_chain[:, :, 0] == -1
#     valid_chain = one_hop_chain[~one_hop_mask]
#     s, r, o, t = valid_chain[:, 0], valid_chain[:, 1], valid_chain[:, 2], valid_chain[:, 3]
#     one_hop_chain_embedding = torch.zeros(B, N, M, entity_embedding.shape[1], device=entity_embedding.device)
#     one_hop_chain_embedding[:, :, :, :] = empty_embedding
#     # pdb.set_trace()
#     valid_chain_embedding = torch.zeros(len(valid_chain), M, entity_embedding.shape[1], device=entity_embedding.device)
#     valid_chain_embedding[:, 0] = entity_embedding[s] + cls_embedding[0]
#     valid_chain_embedding[:, 1] = relation_embedding[r] + cls_embedding[1]
#     valid_chain_embedding[:, 2] = entity_embedding[o] + cls_embedding[2]
#     valid_chain_embedding[:, 3] = time_projection(t.unsqueeze(-1).float()) + cls_embedding[3]
#     unmasked_indices = torch.nonzero(~one_hop_mask, as_tuple=False)  # (K, 2)
#     i_idx, j_idx = unmasked_indices[:, 0], unmasked_indices[:, 1]

#     # 使用批量索引填入
#     one_hop_chain_embedding[i_idx, j_idx] = valid_chain_embedding

#     one_hop_chain_embedding = one_hop_chain_embedding.view(B, -1, entity_embedding.shape[1])
#     return one_hop_chain_embedding, one_hop_mask

def get_embedding(history_chain, entity_embedding, relation_embedding, empty_embedding, time_projection, device='cuda'):
	history_chain = history_chain.to(device)
	history_chain = history_chain.to(torch.int64)
	N, L, M = history_chain.shape[0], history_chain.shape[1], history_chain.shape[2]
	if M != 4:
		raise ValueError('history_chain shape error')

	D = entity_embedding.shape[1]
	chain_embedding = torch.empty(N, L, 4, D).to(entity_embedding.device)
	# pdb.set_trace()
	mask = history_chain[:, :, 0] == -1
	chain = history_chain[~mask]
	s, r, o, t = chain[:, 0], chain[:, 1], chain[:, 2], chain[:, 3]
	chain_embedding = torch.empty(len(chain), 4, D).to(entity_embedding.device)
	chain_embedding[:, 0] = entity_embedding[s]
	chain_embedding[:, 1] = relation_embedding[r]
	chain_embedding[:, 2] = entity_embedding[o]
	chain_embedding[:, 3] = time_projection(t.unsqueeze(-1).float())
	return chain_embedding, mask
# def get_embedding(history_chain, entity_embedding, relation_embedding, empty_embedding, time_projection, device='cuda'):
#     history_chain = history_chain.to(device)
#     history_chain = history_chain.to(torch.int64)
#     N, L, M = history_chain.shape[0], history_chain.shape[1], history_chain.shape[2]
#     if M != 4:
#         raise ValueError('history_chain shape error')

#     D = entity_embedding.shape[1]
#     pdb.set_trace()

#     s = history_chain[:, :, 0]
#     r = history_chain[:, :, 1]
#     o = history_chain[:, :, 2]
#     t = history_chain[:, :, 3]

#     s_mask = s == -1
#     r_mask = r == -1
#     o_mask = o == -1
#     t_mask = t == -1
#     chain_s_embedding = torch.empty((N, L, D)).to(device)
#     chain_r_embedding = torch.empty((N, L, D)).to(device)
#     chain_o_embedding = torch.empty((N, L, D)).to(device)
#     chain_t_embedding = torch.empty((N, L, D)).to(device)
#     chain_s_embedding[s_mask] = empty_embedding
#     chain_r_embedding[r_mask] = empty_embedding
#     chain_o_embedding[o_mask] = empty_embedding
#     chain_t_embedding[t_mask] = empty_embedding
#     chain_s_embedding[~s_mask] = entity_embedding[s[~s_mask]]
#     chain_r_embedding[~r_mask] = relation_embedding[r[~r_mask]]
#     chain_o_embedding[~o_mask] = entity_embedding[o[~o_mask]]
#     # chain_t_embedding[~t_mask] = time_projection(t[~t_mask].unsqueeze(-1).float())
#     chain_embedding[:, :, 0] = chain_s_embedding
#     chain_embedding[:, :, 1] = chain_r_embedding
#     chain_embedding[:, :, 2] = chain_o_embedding
#     chain_embedding[:, :, 3] = chain_t_embedding
	
#     chain_embedding = chain_embedding.view(N, -1, D)
#     return chain_embedding


def score_per_query(score, score_for_query):
	# pdb.set_trace()
	max_score_per_query = torch.zeros_like(score).scatter_reduce(0, score_for_query, score, reduce='amax', include_self=False)
	score_stable = score - max_score_per_query[score_for_query]
	score_stable = torch.exp(score_stable)
	sum_exp_per_query = torch.zeros_like(score).scatter_add(0, score_for_query, score_stable)
	softmax_score = score_stable / sum_exp_per_query[score_for_query]
	return softmax_score


def score_for_query(score, rows_of_false, r=1):
	"""
	:param score: (K,)
	:param rows_of_false: (K,)
	:return:
	"""
	# score: (K,)
	# rows_of_false: (K,)
	# 1. 计算每个 group 的最大值
	# 2. 减去 max，做 exp
	# 3. 对每个 group 累加分母
	# 4. 最终 softmax
	# step 0: prepare
	K = score.shape[0]
	score = score.view(K)  # (K,)
	device = score.device
	unique_rows, inverse_index = torch.unique(rows_of_false, return_inverse=True)  # inverse_index: (K,)
	num_groups = unique_rows.shape[0]

	# step 1: 计算每个 group 的最大值 (for numerical stability)
	max_per_group = torch.full((num_groups,), float('-inf'), device=device)
	max_per_group = max_per_group.scatter_reduce(0, inverse_index, score, reduce='amax', include_self=True)

	# step 2: 减去 max，做 exp
	score_shifted = score - max_per_group[inverse_index]  # (K,)
	score_exp = torch.exp(score_shifted)  # (K,)

	# step 3: 对每个 group 累加分母
	sum_exp_per_group = torch.zeros(num_groups, device=device)
	sum_exp_per_group = sum_exp_per_group.scatter_add(0, inverse_index, score_exp)

	# step 4: 最终 softmax
	score_softmax = score_exp / sum_exp_per_group[inverse_index]
	return score_softmax, inverse_index



class PseudoOntologyDecoder(nn.Module):
	def __init__(self, num_clusters, relation_embedding_dim, entity_embedding_dim):
		super().__init__()
		self.num_clusters = num_clusters
		self.relation_embedding_dim = relation_embedding_dim
		self.entity_embedding_dim = entity_embedding_dim
		
		# 为每个cluster学习一个中心向量，用于soft-type匹配
		self.type_centroids = nn.Parameter(torch.randn(num_clusters, entity_embedding_dim))
		
		# # 关系的“类型模式”：每个关系的头/尾所属伪类型
		# self.relation_head_type = nn.Parameter(torch.randn(num_clusters, relation_embedding_dim))
		# self.relation_tail_type = nn.Parameter(torch.randn(num_clusters, relation_embedding_dim))
		
		# self.scorer = nn.Bilinear(entity_embedding_dim, relation_embedding_dim, 1)

	def generate_pseudo_types(self, entity_embeddings):
		"""
		基于KMeans对实体嵌入聚类，生成伪本体标签（只用于训练初始化）
		"""
		with torch.no_grad():
			kmeans = KMeans(n_clusters=self.num_clusters)
			self.pseudo_labels = kmeans.fit_predict(entity_embeddings.cpu().numpy())
			centroids = torch.tensor(kmeans.cluster_centers_, device=entity_embeddings.device, dtype=entity_embeddings.dtype)
			self.type_centroids.data = centroids
		return self.pseudo_labels

	def forward(self, relation_emb):
		"""
		预测打分，同时引入伪本体的一致性约束
		"""

		match_score = torch.matmul(relation_emb, self.type_centroids.T)  # (B, C)
		# # 计算原始三元组打分
		# triple_score = self.scorer(head_emb, relation_emb).squeeze(-1)

		# # 计算head、tail与每个伪本体中心的相似度（可用于soft类型匹配）
		# head_type_scores = torch.matmul(head_emb, self.type_centroids.T)  # (B, C)
		# tail_type_scores = torch.matmul(tail_emb, self.type_centroids.T)  # (B, C)
		# # 与当前关系的类型模式做匹配（如关系R倾向连接的伪类型）
		# head_match_score = torch.sum(head_type_scores * torch.matmul(relation_emb, self.relation_head_type.T), dim=1)
		# tail_match_score = torch.sum(tail_type_scores * torch.matmul(relation_emb, self.relation_tail_type.T), dim=1)
		# # 综合打分
		# total_score = triple_score + 0.1 * (head_match_score + tail_match_score)
		match_score = match_score[:, self.pseudo_labels]
		return match_score






# class PseudoOntologyDecoder(nn.Module):
#     def __init__(self, num_clusters, relation_embedding_dim, entity_embedding_dim):
#         super().__init__()
#         self.num_clusters = num_clusters
#         self.relation_embedding_dim = relation_embedding_dim
#         self.entity_embedding_dim = entity_embedding_dim
		
#         # 为每个cluster学习一个中心向量，用于soft-type匹配
#         self.type_centroids = nn.Parameter(torch.randn(num_clusters, entity_embedding_dim))
		
#         # 关系的“类型模式”：每个关系的头/尾所属伪类型
#         self.relation_tail_type = nn.Parameter(torch.randn(num_clusters, relation_embedding_dim))
		
#         self.scorer = nn.Bilinear(entity_embedding_dim, relation_embedding_dim, 1)

#     def generate_pseudo_types(self, entity_embeddings):
#         """
#         基于KMeans对实体嵌入聚类，生成伪本体标签（只用于训练初始化）
#         """
#         with torch.no_grad():
#             kmeans = KMeans(n_clusters=self.num_clusters)
#             pseudo_labels = kmeans.fit_predict(entity_embeddings.cpu().numpy())
#             centroids = torch.tensor(kmeans.cluster_centers_, device=entity_embeddings.device, dtype=entity_embeddings.dtype)
#             self.type_centroids.data = centroids
#         return pseudo_labels

#     def forward(self, head_emb, relation_emb, tail_emb):
#         """
#         预测打分，同时引入伪本体的一致性约束
#         """
#         # 计算原始三元组打分
#         triple_score = self.scorer(head_emb, relation_emb).squeeze(-1)

#         # 计算head、tail与每个伪本体中心的相似度（可用于soft类型匹配）
#         head_type_scores = torch.matmul(head_emb, self.type_centroids.T)  # (B, C)
#         tail_type_scores = torch.matmul(tail_emb, self.type_centroids.T)  # (B, C)

#         # 与当前关系的类型模式做匹配（如关系R倾向连接的伪类型）
#         head_match_score = torch.sum(head_type_scores * torch.matmul(relation_emb, self.relation_head_type.T), dim=1)
#         tail_match_score = torch.sum(tail_type_scores * torch.matmul(relation_emb, self.relation_tail_type.T), dim=1)

#         # 综合打分
#         total_score = triple_score + 0.1 * (head_match_score + tail_match_score)
#         return total_score

class ConvTransE(torch.nn.Module):
	def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=3, kernel_size=3, use_bias=True):

		super(ConvTransE, self).__init__()
		# 初始化relation embeddings
		# self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)

		self.inp_drop = torch.nn.Dropout(input_dropout)
		self.hidden_drop = torch.nn.Dropout(hidden_dropout)
		self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)

		self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
							   padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
		self.bn0 = torch.nn.BatchNorm1d(2)
		self.bn1 = torch.nn.BatchNorm1d(channels)
		self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
		self.register_parameter('b',nn.Parameter(torch.zeros(num_entities)))
		self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
		self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
		self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

	def forward(self, embedding, emb_rel, static_emb_ent, head, relation):
		# embedding: entity embeddings (num_entities, embedding_dim)
		# emb_rel: relation embeddings (num_relations, embedding_dim)
		# triplets: (batch_size, 3) 0: head, 1: relation, 2: tail
		# his_emb: historical embeddings (num_entities, embedding_dim)
		# pre_weight: the weight of historical embeddings 0.9
		# pre_type: the type of historical embeddings "all"
		# pdb.set_trace()
		# e1_embedded_all = embedding
		batch_size = len(head)
		e1_embed = F.tanh(embedding).unsqueeze(1)
		el_embedding_all = F.tanh(static_emb_ent)
		rel_embedded = emb_rel[relation].unsqueeze(1)
		stacked_inputs = torch.cat([e1_embed, rel_embedded], 1)
		stacked_inputs = self.bn0(stacked_inputs)
		x = self.inp_drop(stacked_inputs)
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.feature_map_drop(x)
		x = x.view(batch_size, -1)
		x = self.fc(x)
		x = self.hidden_drop(x)
		if batch_size > 1:
			x = self.bn2(x)
		x = F.relu(x)
		cl_x = x
		x = torch.mm(x, el_embedding_all.transpose(1, 0))
		return x, cl_x


def masked_mean_pooling_original(chain_embedding, mask):
	B, N, group, D = mask.shape[0], mask.shape[1], 4, chain_embedding.shape[2]  # 128, 25, 4, 256
	chain_embedding = chain_embedding.view(B, N, group, D)  # (128, 25, 4, 256)

	# 反转 mask，False 表示无效 → True，True 表示有效 → False
	valid_mask = ~mask  # shape (128, 25)

	# 扩展 mask 用于广播到 embedding
	expanded_mask = valid_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, group, D)

	# 取有效部分
	valid_embeddings = chain_embedding * expanded_mask

	# 计算有效 token 数量
	num_valid_tokens = valid_mask.sum(dim=1).clamp(min=1) * group  # shape: (128,)

	# 默认平均
	pooled = valid_embeddings.sum(dim=1).sum(dim=1) / num_valid_tokens.unsqueeze(-1)  # (128, 256)

	# 找出哪些样本完全无效（mask 全为 True）
	all_invalid = ~valid_mask.any(dim=1)  # shape (128,), True 表示该样本全是 masked

	# 用 chain_embedding 第一个 token 替代这些行（从 group 展平后的 index 0）
	fallback = chain_embedding.view(B, N * group, D)[:, 0, :]  # shape: (128, 256)

	# 替换全为 True 的样本
	pooled[all_invalid] = fallback[all_invalid]

	return pooled


def masked_mean_pooling(chain_embedding, mask):
	B, N, D = mask.shape[0], mask.shape[1], chain_embedding.shape[2]  # 128, 25, 4, 256
	chain_embedding = chain_embedding.view(B, N, D)  # (128, 25, 4, 256)

	# 反转 mask，False 表示无效 → True，True 表示有效 → False
	valid_mask = ~mask  # shape (128, 25)
	# pdb.set_trace()
	# 扩展 mask 用于广播到 embedding
	expanded_mask = valid_mask.unsqueeze(-1).expand(-1, -1, D)

	# 取有效部分
	valid_embeddings = chain_embedding * expanded_mask

	# 计算有效 token 数量
	num_valid_tokens = valid_mask.sum(dim=1).clamp(min=1)  # shape: (128,)

	# 默认平均
	pooled = valid_embeddings.sum(dim=1) / num_valid_tokens.unsqueeze(-1)  # (128, 256)

	# 找出哪些样本完全无效（mask 全为 True）
	all_invalid = ~valid_mask.any(dim=1)  # shape (128,), True 表示该样本全是 masked

	# 用 chain_embedding 第一个 token 替代这些行（从 group 展平后的 index 0）
	fallback = chain_embedding.view(B, N, D)[:, 0, :]  # shape: (128, 256)

	# 替换全为 True 的样本
	pooled[all_invalid] = fallback[all_invalid]

	return pooled


class CodebookDecoder(nn.Module):
	def __init__(self, num_codes, embedding_dim, commitment_beta=0.25):
		super().__init__()
		self.num_codes = num_codes
		self.embedding_dim = embedding_dim
		self.codebook = nn.Parameter(torch.randn(num_codes, embedding_dim))  # 伪类型向量
		self.commitment_beta = commitment_beta

	def quantize(self, x):
		"""
		向量量化（最近邻查找）+ 计算VQ损失
		"""
		# x: (B, D), codebook: (C, D)
		x_expanded = x.unsqueeze(1)  # (B, 1, D)
		codebook_expanded = self.codebook.unsqueeze(0)  # (1, C, D)

		distances = torch.norm(x_expanded - codebook_expanded, dim=2)  # (B, C)
		nearest_idx = torch.argmin(distances, dim=1)  # (B,)
		quantized = self.codebook[nearest_idx]  # (B, D)

		# VQ loss
		vq_loss = F.mse_loss(quantized.detach(), x) + self.commitment_beta * F.mse_loss(quantized, x.detach())
		return quantized, nearest_idx, vq_loss

	def forward(self, query_emb, entity_emb):
		# 量化得到伪类型 + loss
		quant, head_idx, vq_loss = self.quantize(entity_emb)

		score = torch.matmul(query_emb, quant.T)

		return score, vq_loss, head_idx

class VectorQuantizerEMA(nn.Module):
	def __init__(self, num_codes, embedding_dim, commitment_beta=0.25, decay=0.99, usage_lambda=5e-3):
		super().__init__()
		self.num_codes = num_codes
		self.embedding_dim = embedding_dim
		self.commitment_beta = commitment_beta
		self.decay = decay
		self.usage_lambda = usage_lambda

		codebook = torch.randn(num_codes, embedding_dim) * 0.1
		self.register_buffer('codebook', codebook)               # 不对其反传
		self.register_buffer('ema_count', torch.zeros(num_codes))
		self.register_buffer('ema_sum', torch.zeros(num_codes, embedding_dim))

	@torch.no_grad()
	def _update_ema(self, x, idx):
		# x: (B,D)
		one_hot = F.one_hot(idx, num_classes=self.num_codes).type_as(self.ema_count)  # (B,C)
		count = one_hot.sum(dim=0)                          # (C,)
		sum_vec = one_hot.t() @ x                           # (C,D)

		self.ema_count.mul_(self.decay).add_(count, alpha=1-self.decay)
		self.ema_sum.mul_(self.decay).add_(sum_vec, alpha=1-self.decay)

		n = self.ema_count + 1e-5
		self.codebook.copy_( self.ema_sum / n.unsqueeze(-1) )

		# 复活 dead codes
		dead = (self.ema_count < 1e-3).nonzero(as_tuple=True)[0]
		if len(dead) > 0:
			r = torch.randn(len(dead), self.embedding_dim, device=self.codebook.device)
			r = F.normalize(r, dim=-1) * 0.1
			self.codebook[dead] = r
			self.ema_sum[dead] = r
			self.ema_count[dead] = 1.0

	def forward(self, x):
		# 归一化 + 余弦距离
		x_n = F.normalize(x, dim=-1)
		cb_n = F.normalize(self.codebook, dim=-1)
		sims = x_n @ cb_n.t()
		distances = 1.0 - sims
		idx = distances.argmin(dim=1)
		quant = self.codebook[idx]

		# VQ 损失（只对 x 侧回传）
		# codebook 不反传，commitment 惩罚就够
		commit_loss = self.commitment_beta * F.mse_loss(x, quant.detach())

		# 使用度熵正则
		with torch.no_grad():
			counts = torch.bincount(idx, minlength=self.num_codes).float()
		p = counts / (counts.sum() + 1e-6)
		entropy = -(p * (p + 1e-12).log()).sum()

		logC = torch.log(
			torch.tensor(float(self.num_codes), device=entropy.device, dtype=entropy.dtype) + 1e-12
		)
		usage_loss = self.usage_lambda * (logC - entropy)
		vq_loss = commit_loss + usage_loss

		# usage_loss = self.usage_lambda * ((self.num_codes + 1e-12).log() - entropy)

		# vq_loss = commit_loss + usage_loss

		# EMA 更新
		self._update_ema(x.detach(), idx)

		aux = {"usage_entropy": entropy.detach(),
			   "perplexity": entropy.exp().detach(),
			   "counts": counts.detach()}
		return quant, idx, vq_loss, aux




class BasicDriftMLP(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim*2, embed_dim),
			nn.ReLU(),
			nn.Linear(embed_dim, embed_dim)
		)
		# self.norm = nn.LayerNorm(embed_dim)

	def forward(self, pseudo_event_emb):
		"""
		输入: pseudo_event_emb (B, D) - 每个实体所属本体类别在当前时间的伪事件表示
		输出: delta (B, D) - 表征漂移量
		"""
		delta = self.mlp(pseudo_event_emb)
		return delta



class MyModel(nn.Module):
	def __init__(self,
				num_ent,
				num_rel,
				num_heads,
				entity_dim,
				relation_dim,
				num_layers,
				dropout=0.0,
				word_embedding_path=None,
				word_embedding=False,
				residual=True,
				device='cuda',
				layer_norm=False,
				chain_max_length=10,
				time_length=14,
				word_embedding_dim=768,
				num_code=50,
				ablation=None
				):
		super(MyModel, self).__init__()
		self.activation = F.relu
		self.word_embedding = word_embedding
		self.residual = residual
		self.num_rel = num_rel
		self.num_ents = num_ent
		num_rel = num_rel * 2
		self.layer_norm = layer_norm
		self.weight_t = nn.Parameter(torch.randn(1, entity_dim))
		self.bias_t = nn.Parameter(torch.randn(1, entity_dim))
		self.device = device
		self.initializer_range = 0.02
		self.bn_entity = torch.nn.BatchNorm1d(entity_dim)
		self.bn_relation = torch.nn.BatchNorm1d(relation_dim)
		self.entity_dim = entity_dim
		self.chain_max_length = chain_max_length
		self.bn_1 = torch.nn.BatchNorm1d(entity_dim)
		self.bn_2 = torch.nn.BatchNorm1d(entity_dim)
		self.entity_down_proj = nn.Linear(word_embedding_dim, int(entity_dim/4))
		self.relation_down_proj = nn.Linear(relation_dim, int(relation_dim/4))
		self.time_projection = nn.Linear(1, int(entity_dim/4))
		self.num_code = num_code
		self.time_length = time_length
		if self.word_embedding:
			self.entity_embedding = torch.tensor(np.load(word_embedding_path), dtype=torch.float).to('cuda')
		if self.word_embedding and (entity_dim != word_embedding_dim):
			self.project = nn.Linear(word_embedding_dim, entity_dim)
		else:
			self.project = nn.Linear(entity_dim, entity_dim)
		self.relation_embedding = nn.Parameter(torch.Tensor(num_rel, relation_dim)).to(device)
		self.empty_embedding = nn.Parameter(torch.Tensor(1, entity_dim)).to(device)
		self.cls_embedding = nn.Parameter(torch.Tensor(4, entity_dim)).to(device)

		nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.empty_embedding, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.cls_embedding, gain=nn.init.calculate_gain('relu'))


		self.filling_embedding = nn.Parameter(torch.Tensor(1, entity_dim)).to(device)
		nn.init.xavier_uniform_(self.filling_embedding, gain=nn.init.calculate_gain('relu'))

		self.merge_layer = nn.Linear(entity_dim+relation_dim, entity_dim)
		self.encoder = TransformerEncoder_with_query(num_layers, entity_dim, num_heads, entity_dim * 2, dropout)
		self.w = nn.Linear(self.entity_dim*2, self.entity_dim)
		self.w2 = nn.Linear(self.entity_dim, 1)
		self.w4 = nn.Linear(self.entity_dim*2, 1)
		self.scoring_layer1 = nn.Linear(self.entity_dim, self.entity_dim)
		self.scoring_layer2 = nn.Linear(self.entity_dim, 1)
		self.projection = nn.Linear(entity_dim*2, entity_dim)
		self.relation_proj = nn.Linear(relation_dim, relation_dim)
		self.lstm_encoder = nn.LSTM(entity_dim, entity_dim, batch_first=True)
		self.mlp_encoder_1 = nn.Linear(entity_dim*4, entity_dim*2)
		self.mlp_encoder_2 = nn.Linear(entity_dim*2, entity_dim)
		self.decoder = ConvTransE(num_ent, entity_dim, input_dropout=dropout, hidden_dropout=dropout, feature_map_dropout=dropout)
		self.VQDecoder = CodebookDecoder(num_codes=self.num_code, embedding_dim=entity_dim)
		self.VQ = VectorQuantizerEMA(num_codes=self.num_code, embedding_dim=entity_dim, commitment_beta=0.25, decay=0.99, usage_lambda=5e-3)

		self.Drift = BasicDriftMLP(entity_dim)
		self.ablation = ablation
		# self.pseudo_labels = self.cluster.generate_pseudo_types(entity_embedding.to('cuda'))
		print('init model')
		print('entity_embedding', self.entity_embedding.shape)
		print('relation_embedding', self.relation_embedding.shape)

	
	def forward(self, triples, chain_embedding, chain_mask, embedding_dict):
		torch.autograd.set_detect_anomaly(True)
		ground_truth = triples
		query_relation_embedding = embedding_dict['relation_embedding'][ground_truth[:, 1]]
		# pdb.set_trace()
		if self.ablation == 'no_ITC':
			query_embedding = embedding_dict['entity_embedding'][ground_truth[:, 0]]
		else:
			chain_embedding = self.encoder(chain_embedding)
			chain_weight_embedding = chain_embedding + query_relation_embedding.unsqueeze(1)
			chain_weight_embedding = chain_weight_embedding.reshape(-1, self.entity_dim)
			att = F.softmax(self.w2(chain_weight_embedding).reshape(chain_embedding.shape[0], -1), dim=1)
			chain_embedding_flatten = att.unsqueeze(-1) * chain_embedding
			chain_embedding = torch.mean(chain_embedding_flatten, dim=1)
			if self.ablation == 'no_batchnorm':
				query_embedding = chain_embedding
			else:
				query_embedding = self.bn_1(chain_embedding)
		query_entity = ground_truth[:, 0]
		edge_type = ground_truth[:, 1]
		label = ground_truth[:, 2]

		relation = self.bn_relation(query_relation_embedding)
		query_embedding = query_embedding + relation
		static_entity_embedding = self.project(self.entity_embedding)
		max_entity_index = query_entity.max().item() + 1
		if self.ablation == 'ITC':
			static_entity_embedding = embedding_dict['entity_embedding'][ground_truth[:, 0]]
		else:
			static_entity_embedding[query_entity] = static_entity_embedding[query_entity] + query_embedding
		# VQ_score, vq_loss, pseudo_onto_idx = self.VQDecoder(relation, static_entity_embedding)
		quant, pseudo_onto_idx, vq_loss, aux = self.VQ(static_entity_embedding)     # entity_emb: (N,D) 静态候选实体嵌入
		# score = query_emb @ quant.t()                 # (B,N)
		# loss = lp_loss + vq_loss
		query_onto = pseudo_onto_idx[query_entity]
		entity_onto_embedding = static_entity_embedding[pseudo_onto_idx]

		drift_embedding = torch.zeros(self.num_code, self.entity_dim).to(self.device)
		drift_embedding.index_add_(0, query_onto, query_embedding)
		count = torch.bincount(query_onto, minlength=self.num_code).unsqueeze(-1).clamp(min=1)
		drift_embedding = drift_embedding / count

		drift_embedding = drift_embedding[pseudo_onto_idx]

		drift_query_embedding = torch.cat((static_entity_embedding, entity_onto_embedding), dim=1)
		drift_weight = self.Drift(drift_query_embedding)

		query_entity_unique = query_entity.unique() 
		not_query_entity = torch.tensor([i for i in range(self.num_ents) if i not in query_entity_unique])
		dynamic_entity_embedding = static_entity_embedding.clone()
		## no_drift or no_codebook

		if self.ablation in ['no_drift', 'no_codebook']:
			dynamic_entity_embedding = self.project(self.entity_embedding)
		else:
			dynamic_entity_embedding[not_query_entity] = dynamic_entity_embedding[not_query_entity] + (drift_weight * drift_embedding)[not_query_entity]

		dynamic_entity_embedding = F.normalize(dynamic_entity_embedding) if self.layer_norm else dynamic_entity_embedding



		scores_ob, _= self.decoder.forward(query_embedding, self.relation_embedding, dynamic_entity_embedding, query_entity, edge_type)
		# print(scores_ob.isnan().any())
		
		# print(scores_ob.isnan().any())
		if self.ablation == 'no_codebook':
			scores_en = F.log_softmax(scores_ob, dim=1)
			loss = F.nll_loss(scores_en, label)
			return scores_en, loss
		# scores_ob = scores_ob + VQ_score
		scores_en = F.log_softmax(scores_ob, dim=1)

		loss = F.nll_loss(scores_en, label) + 0.1*vq_loss

		return scores_en, loss

