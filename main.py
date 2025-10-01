import os
import numpy as np
import torch
import argparse
from model import *
import utils
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import wandb
import logging
from datetime import datetime
import sys
from tqdm import tqdm
import pickle
import random
class PrintToLog:
    def write(self, message):
        if message != '\n':  # 忽略空行
            logging.info(message)

    def flush(self):
        pass  # 可选，实现flush方法

def setup_logging(log_file):
    # Set up logging configuration
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a handler to output logs to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

sys.stdout = PrintToLog()

def create_timestamped_dir(base_dir, args):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(base_dir, timestamp)
    results_dir = results_dir + f"_dataset_{args.dataset}_history_len_{args.history_len}_max_length_{args.max_length}_hidden_dim_{args.hidden_dim}_num_layers_{args.num_layers}_num_heads_{args.num_heads}_num_codes_{args.num_code}_ratio_{args.split_ratio}_tips_{args.tips}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir



    
def re_spilt(dataset, split_ratio=[0.5, 0.2, 0.3]):
    all_triples = np.concatenate([dataset.train, dataset.valid, dataset.test], axis=0)
    # np.random.shuffle(all_data)m ;p0
    times = np.unique(all_triples[:, 3])
    train_time = times[:int(len(times)*split_ratio[0])]
    valid_time = times[int(len(times)*split_ratio[0]):int(len(times)*(split_ratio[0]+split_ratio[1]))]
    test_time = times[int(len(times)*(split_ratio[0]+split_ratio[1])):]
    train_triples = all_triples[all_triples[:, 3]<=train_time[-1], :]
    val_triples = all_triples[(all_triples[:, 3]>train_time[-1])&(all_triples[:, 3]<=valid_time[-1])]
    test_triples = all_triples[(all_triples[:, 3]>valid_time[-1])&(all_triples[:, 3]<=test_time[-1])]
    dataset.train = train_triples
    dataset.valid = val_triples
    dataset.test = test_triples
    return dataset


def find_unknown_triple_index(triple, unkown_entity_index):
    all_triples = np.concatenate(triple, axis=0)
    in_triple = np.isin(all_triples[:, [0, 2]], list(unkown_entity_index))
    unknown_index = np.where(np.any(in_triple, axis=1))[0]
    return unknown_index

class TKGDataset(Dataset):
    def __init__(self, data_dict, all_triples):
        self.data_dict = data_dict
        self.triples = all_triples
        self.times = np.unique(all_triples[:, 3])
        self.num_relations = np.unique(all_triples[:, 1]).shape[0]
        self.num_entities = np.unique(all_triples[:, [0, 2]]).shape[0]
    def __len__(self):
        return len(self.times)
    
    def __getitem__(self, idx):
        time = self.times[idx]
        triples_at_time = self.triples[self.triples[:, 3] == time]
        return torch.tensor(triples_at_time)

 
def prepare_data(ground_truth, data, one_hop_length=30, two_hop_length=100):
    length = len(data)
    one_hop_chain_all = np.zeros((length, one_hop_length, 4))-1
    two_hop_chain_all = np.zeros((length, two_hop_length, 8))-1
    for idx in range(length):
        one_hop_chain = data[idx]['one_hop_chain']
        two_hop_chain = data[idx]['two_hop_chain']
        if len(one_hop_chain) > one_hop_length:
            random_index = np.random.randint(0, len(one_hop_chain), size=one_hop_length)
            one_hop_chain = one_hop_chain[random_index]
        if len(two_hop_chain) > two_hop_length:
            random_index = np.random.randint(0, len(two_hop_chain), size=two_hop_length)
            two_hop_chain = two_hop_chain[random_index]
        one_hop_chain_all[idx, :len(one_hop_chain), :4] = one_hop_chain
        one_hop_chain_all[idx, :len(one_hop_chain), 3] = ground_truth[idx, 3] - one_hop_chain_all[idx, :len(one_hop_chain), 3]
        two_hop_chain_all[idx, :len(two_hop_chain), :] = two_hop_chain
        two_hop_chain_all[idx, :len(two_hop_chain), 3] = ground_truth[idx, 3] - two_hop_chain_all[idx, :len(two_hop_chain), 3]
    return ground_truth, torch.tensor(one_hop_chain_all), torch.tensor(two_hop_chain_all)

def get_topk_chain_through_relation_embedding(query_relation, chain_relation, relation_embedding, topk=30):
    query_relation_embedding = relation_embedding[query_relation]
    chain_relation_embedding = relation_embedding[chain_relation]
    query_relation_embedding = query_relation_embedding.unsqueeze(0)
    similarity = torch.matmul(query_relation_embedding, chain_relation_embedding.T)
    _, topk_index = torch.topk(similarity, topk)
    return topk_index
    

def get_embedding_only_one_hop(head, one_hop_chain, embedding_dict, model, device='cuda'):
    entity_embedding, relation_embedding, cls_embedding, empty_embedding = embedding_dict['entity_embedding'], embedding_dict['relation_embedding'], embedding_dict['cls_embedding'], embedding_dict['missing_embedding']
    B, N, M = one_hop_chain.shape[0], one_hop_chain.shape[1], one_hop_chain.shape[2]
    one_hop_chain = one_hop_chain.to(torch.int64)
    one_hop_chain = sort_by_last_dim_with_neg1_last(one_hop_chain)
    time_projection = model.time_projection
    entity_proj = model.entity_down_proj
    relation_proj = model.relation_down_proj
    one_hop_mask = one_hop_chain[:, :, 0] == -1
    valid_chain = one_hop_chain[~one_hop_mask]
    s, r, o, t = valid_chain[:, 0], valid_chain[:, 1], valid_chain[:, 2], valid_chain[:, 3]
    one_hop_chain_embedding = torch.zeros(B, N, M, int(relation_embedding.shape[1]/M), device=relation_embedding.device)
    one_hop_chain_embedding[:, :, :, :] = empty_embedding[:, int(relation_embedding.shape[1]/M)]
    valid_chain_embedding = torch.zeros(len(valid_chain), M, int(relation_embedding.shape[1]/4), device=relation_embedding.device)
    valid_chain_embedding[:, 0] = entity_proj(entity_embedding[s])
    valid_chain_embedding[:, 1] = relation_proj(relation_embedding[r])
    valid_chain_embedding[:, 2] = entity_proj(entity_embedding[o])
    valid_chain_embedding[:, 3] = time_projection(t.unsqueeze(-1).float())
    unmasked_indices = torch.nonzero(~one_hop_mask, as_tuple=False)  # (K, 2)
    i_idx, j_idx = unmasked_indices[:, 0], unmasked_indices[:, 1]
    one_hop_chain_embedding[:, 0, 0] = entity_proj(entity_embedding[head])
    one_hop_chain_embedding[i_idx, j_idx] = valid_chain_embedding
    one_hop_chain_embedding = one_hop_chain_embedding.view(B, -1, relation_embedding.shape[1])
    return one_hop_chain_embedding, one_hop_mask


def prepare_data_one_hop_chain(ground_truth, data, embedding_dict, model, one_hop_length=30):
    length = len(data)
    one_hop_chain_all = np.zeros((length, one_hop_length, 4))-1
    entity_embedding, relation_embedding, cls_embedding, missing_embedding = embedding_dict['entity_embedding'], embedding_dict['relation_embedding'], embedding_dict['cls_embedding'], embedding_dict['missing_embedding']
    for idx in range(length):
        one_hop_chain = data[idx]['one_hop_chain']
        if len(one_hop_chain) > one_hop_length:
            query_relation = ground_truth[idx, 1]
            chain_relation = one_hop_chain[:, 1]
            topk_index = get_topk_chain_through_relation_embedding(query_relation, chain_relation, relation_embedding, topk=one_hop_length)
            one_hop_chain = one_hop_chain[topk_index.squeeze().detach().cpu().numpy()]
        one_hop_chain_all[idx, :len(one_hop_chain), :4] = one_hop_chain
        one_hop_chain_all[idx, :len(one_hop_chain), 3] = ground_truth[idx, 3] - one_hop_chain_all[idx, :len(one_hop_chain), 3]
    head = ground_truth[:, 0]
    
    chain_embedding, chain_mask = get_embedding_only_one_hop(head, torch.tensor(one_hop_chain_all).to('cuda'), embedding_dict, model, device='cuda')
    return chain_embedding, chain_mask, one_hop_chain_all

def get_init_embedding(entity_embedding_path, n_entity, n_relation, hidden_dim, device='cuda', word_embedding=True):
    gamma = 6.0
    epsilon = 1.0
    embedding_range = nn.Parameter(
                torch.Tensor([(gamma + epsilon) / hidden_dim]),
                requires_grad=False
            )
    if word_embedding:
        entity_embedding = torch.tensor(np.load(entity_embedding_path), dtype=torch.float).to('cuda')
    else:
        entity_embedding = nn.Parameter(torch.zeros(n_entity, hidden_dim, device=device))
        nn.init.uniform_(
            tensor=entity_embedding,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
    relation_embedding = nn.Parameter(torch.zeros(n_relation*2, hidden_dim, device=device))
    nn.init.uniform_(
        tensor=relation_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    cls_embedding = nn.Parameter(torch.zeros(4, hidden_dim, device=device))
    missing_embedding = nn.Parameter(torch.zeros((1, hidden_dim), device=device))
    nn.init.uniform_(
        tensor=cls_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    nn.init.uniform_(
        tensor=missing_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    embedding_dict = {}
    embedding_dict['entity_embedding'] = entity_embedding
    embedding_dict['relation_embedding'] = relation_embedding
    embedding_dict['cls_embedding'] = cls_embedding
    embedding_dict['missing_embedding'] = missing_embedding
    return embedding_dict

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Knowledge Graph Embedding")
    parser.add_argument("--dataset", type=str, default="ICEWS14", help="dataset")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--history_len", type=int, default=14, help="train history length")
    parser.add_argument("--max_length", type=int, default=30, help="max length of thechain")
    parser.add_argument("--hidden_dim", type=int, default=768, help="hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--word_embedding", action='store_false', default=True,  help="word embedding")
    parser.add_argument("--word_embedding_path", type=str, default='data', help="word embedding path")
    parser.add_argument("--word_embedding_dim", type=int, default=768, help="word embedding dimension")
    parser.add_argument("--residual", type=bool, default=True, help="residual")
    parser.add_argument("--result_dir", type=str, default='results', help="result dir")
    parser.add_argument("--layer_norm", action='store_false', default=True, help="layer norm")
    parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
    parser.add_argument("--tips", type=str, default='None', help="use tips")
    parser.add_argument("--patience", type=int, default=10, help="patience for early stopping")
    parser.add_argument("--seed", type=int, nargs='+', default=[42], help="list of random seeds")
    parser.add_argument("--num_code", type=int, default=50, help="number of clusters for clustering")
    parser.add_argument("--ablation", type=str, default='None', help="ablation study")
    parser.add_argument("--split_ratio", type=int, default=30, help="train/val/test split ratio")
    return parser.parse_args()

def main(args):
    if args.ablation == 'None':
        results_dir = create_timestamped_dir(args.result_dir, args)
    else:
        args.result_dir = 'ablation_results/' + args.ablation
        results_dir = create_timestamped_dir(args.result_dir, args)
    setup_logging(os.path.join(results_dir, 'log.txt'))
    if args.ablation == 'no_codebook':
        args.num_code = 1
    args.word_embedding_path = args.word_embedding_path + f'/{args.dataset}/{args.dataset}_Bert_Entity_Embedding.npy'
    print(args)
    gpu_id = args.gpu
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print("loading graph data")
    data = utils.load_data(args.dataset)
    ratio_name = {10: [0.8, 0.1, 0.1], 30: [0.5, 0.2, 0.3], 50: [0.3, 0.2, 0.5], 70: [0.2, 0.1, 0.7]}
    split_ratio= ratio_name[args.split_ratio]
    data = re_spilt(data, split_ratio)
    known_entity = set(data.train[:, 0].tolist() + data.train[:, 2].tolist())
    known_entity_index = torch.tensor(list(known_entity), dtype=torch.long)
    print("Number of known entities: ", len(known_entity))
    unknown_entity = set(range(data.num_nodes)) - known_entity
    print("Number of unknown entities: ", len(unknown_entity))
    unkown_entity_index = torch.tensor(list(unknown_entity), dtype=torch.long)

    known_entity_train_val = set(data.train[:, 0].tolist() + data.train[:, 2].tolist() + data.valid[:, 0].tolist() + data.valid[:, 2].tolist())
    known_entity_train_val_index = torch.tensor(list(known_entity_train_val), dtype=torch.long)
    unkown_entity_test = set(range(data.num_nodes)) - known_entity_train_val
    unkown_entity_test_index = torch.tensor(list(unkown_entity_test), dtype=torch.long)
    train_triple = data.train
    valid_triple = data.valid
    test_triple = data.test
    all_triple = np.concatenate([train_triple, valid_triple, test_triple], axis=0)
    times = np.unique(all_triple[:, 3])
    train_time = times[:int(len(times)*split_ratio[0])]
    valid_time = times[int(len(times)*split_ratio[0]):int(len(times)*(split_ratio[0]+split_ratio[1]))]
    test_time = times[int(len(times)*(split_ratio[0]+split_ratio[1])):]
    test_time = test_time[:-1]

    all_entities = np.concatenate([all_triple[:, 0], all_triple[:, 2]])
    all_entities = np.unique(all_entities)
    all_relations = np.unique(all_triple[:, 1])
    all_times = np.unique(all_triple[:, 3])
    all_triple_inverse = all_triple[:, [2, 1, 0, 3]]    
    all_triple_inverse[:, 1] += len(all_relations)
    all_triple = np.concatenate([all_triple, all_triple_inverse], axis=0)
    entity_history = {}
    for entity in all_entities:
        triple_with_entity = all_triple[all_triple[:, 0] == entity]
        triple_with_entity_sort_by_time = triple_with_entity[np.argsort(triple_with_entity[:, 3])]
        entity_history[entity] = triple_with_entity_sort_by_time

    dataset = TKGDataset(entity_history, all_triple)
    
    file_path = f'data/{args.dataset}/{args.dataset}_T_{args.history_len}.pkl'
    with open(file_path, 'rb') as f:
        hisotry_dataset = pickle.load(f)
        
    best_result_dict = {}
    for seed in args.seed:
        set_random_seed(seed)
        print(f"Running with seed {seed}")
        if args.ablation in ['no_word_embedding', 'no_ITC']:
            embedding_dict = get_init_embedding(args.word_embedding_path, data.num_nodes, data.num_rels, args.hidden_dim, device=device, word_embedding=False)
        else:
            embedding_dict = get_init_embedding(args.word_embedding_path, data.num_nodes, data.num_rels, args.hidden_dim, device=device)
        model = MyModel(data.num_nodes, data.num_rels, num_heads = args.num_heads, entity_dim=args.hidden_dim, relation_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, word_embedding=args.word_embedding, word_embedding_path=args.word_embedding_path, layer_norm=args.layer_norm, word_embedding_dim = args.word_embedding_dim, num_code = args.num_code, ablation=args.ablation).to(device)
        if args.ablation in ['no_word_embedding']:
            all_params = list(model.parameters()) + [embedding_dict['relation_embedding'], embedding_dict['cls_embedding'], embedding_dict['missing_embedding'], embedding_dict['entity_embedding']]
        else:
            all_params = list(model.parameters()) + [embedding_dict['relation_embedding'], embedding_dict['cls_embedding'], embedding_dict['missing_embedding']]
        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)
        best_val_loss = 100000
        patience = 0
        max_patience = args.patience
        model_name = f"model_dataset_{args.dataset}_history_len_{args.history_len}_max_length_{args.max_length}_hidden_dim_{args.hidden_dim}_num_layers_{args.num_layers}_num_heads_{args.num_heads}_num_codes_{args.num_code}_tips_{args.tips}_seed_{seed}.pth"
        best_val_mrr = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            train_loss_list = []
            rank_list = []
            unvalid_ratio = 0

            for year in tqdm(range(len(train_time))):
                data_for_year = dataset[year]
                history_data = hisotry_dataset[year]
                for batch_id in range(2):
                    start_id = int(batch_id * len(data_for_year) / 2)
                    end_id = int((batch_id + 1) * len(data_for_year)/ 2)
                    chain_embedding, chain_mask, _ = prepare_data_one_hop_chain(data_for_year, history_data, embedding_dict, model, one_hop_length=args.max_length)
                    triples = data_for_year[start_id:end_id]
                    chain_embedding = chain_embedding[start_id:end_id]
                    chain_mask = chain_mask[start_id:end_id]

                    triples = triples.to(device)
                    score, loss = model(triples, chain_embedding, chain_mask, embedding_dict)
                    # print(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    train_loss += loss.item()
                    train_loss_list.append(loss.item())
                    rank = utils.get_rank(score, triples[:, 2])
                    rank_list.append(rank)
            unvalid_ratio /= len(train_time)
            mrr, hit1, hit3, hit10 = utils.get_metric(rank_list)          
            
            print(f"Epoch: {epoch}, Train Loss: {train_loss/len(train_loss_list):.4f}")
            print(f"MRR: {mrr:.4f}, Hit1: {hit1:.4f}, Hit3: {hit3:.4f}, Hit10: {hit10:.4f}")

            model.eval()
            valid_loss = 0

            valid_loss_list = []
            rank_list = []
            inv_rank_list = []
            valid_triples = []
            unvalid_ratio = 0
            with torch.no_grad():
                select_chains = np.zeros((0, args.max_length, 4))
                for year in range(len(train_time), len(train_time) + len(valid_time)):
                    data_for_year = dataset[year]
                    history_data = hisotry_dataset[year]
                    batch_triples = np.zeros((0, 4))
                    for batch_id in range(2):
                        start_id = int(batch_id * len(data_for_year) / 2)
                        end_id = int((batch_id + 1) * len(data_for_year) / 2)
                        chain_embedding, chain_mask, select_chain= prepare_data_one_hop_chain(data_for_year, history_data, embedding_dict, model, one_hop_length=args.max_length)
                        triples = data_for_year[start_id:end_id]
                        chain_embedding = chain_embedding[start_id:end_id]
                        chain_mask = chain_mask[start_id:end_id]
                        select_chain = select_chain[start_id:end_id]
                        triples = triples.to(device)
                        score, loss = model(triples, chain_embedding, chain_mask, embedding_dict)
                        valid_loss += loss.item()
                        valid_loss_list.append(loss.item())
                        rank = utils.get_rank(score, triples[:, 2])
                        rank_list.append(rank.cpu())
                        batch_triples = np.concatenate((batch_triples, triples.cpu().numpy()), axis=0)
                        select_chains = np.concatenate((select_chains, select_chain), axis=0)
                    valid_triples.append(batch_triples)
                
                unvalid_ratio /= len(valid_time)
                mrr, hit1, hit3, hit10 = utils.get_metric(rank_list)
                if epoch == 0:
                    valid_unknown_triple_index = utils.get_unkown_index(valid_triples, unkown_entity_index)
                    valid_emerging_triple_index = utils.get_emerging_index(valid_triples, known_entity_index)
                    print(f"Unknown triples in valid set: {len(valid_unknown_triple_index)}, Emerging triples in valid set: {len(valid_emerging_triple_index)}")
                unknown_mrr_valid, unknown_hit1, unknown_hit3, unknown_hit10 = utils.get_metric_unknown_both(rank_list, valid_triples, valid_unknown_triple_index)
                emerging_mrr_valid, emerging_hit1, emerging_hit3, emerging_hit10 = utils.get_metric_emerging_both(rank_list, valid_triples, valid_emerging_triple_index)
                print(f"Epoch: {epoch}, Valid Loss: {valid_loss/len(valid_loss_list):.4f}")
                print(f"MRR: {mrr:.4f}, Hit1: {hit1:.4f}, Hit3: {hit3:.4f}, Hit10: {hit10:.4f}")
                print(f"Unknown_MRR: {unknown_mrr_valid:.4f}, Unknown_Hit1: {unknown_hit1:.4f}, Unknown_Hit3: {unknown_hit3:.4f}, Unknown_Hit10: {unknown_hit10:.4f}")
                print(f"Emerging_MRR: {emerging_mrr_valid:.4f}, Emerging_Hit1: {emerging_hit1:.4f}, Emerging_Hit3: {emerging_hit3:.4f}, Emerging_Hit10: {emerging_hit10:.4f}")

            if emerging_mrr_valid > best_val_mrr:
                best_val_mrr = emerging_mrr_valid
                test_loss = 0
                test_loss_list = []
                rank_list = []
                inv_rank_list = []
                test_triples = []
                unvalid_ratio = 0
                select_chains = np.zeros((0, args.max_length, 4))
                with torch.no_grad():
                    for year in range(len(train_time) + len(valid_time), len(train_time) + len(valid_time) + len(test_time)-1):
                        data_for_year = dataset[year]
                        ## 按batch 划分
                        history_data = hisotry_dataset[year]
                        batch_triples = np.zeros((0, 4))
                        for batch_id in range(2):
                            start_id = int(batch_id * len(data_for_year) / 2)
                            end_id = int((batch_id + 1) * len(data_for_year) / 2)
                            chain_embedding, chain_mask, select_chain= prepare_data_one_hop_chain(data_for_year, history_data, embedding_dict, model, one_hop_length=args.max_length)
                            triples = data_for_year[start_id:end_id]
                            chain_embedding = chain_embedding[start_id:end_id]
                            chain_mask = chain_mask[start_id:end_id]
                            select_chain = select_chain[start_id:end_id]
                            triples = triples.to(device)
                            score, loss = model(triples, chain_embedding, chain_mask, embedding_dict)
                            test_loss += loss.item()
                            test_loss_list.append(loss.item())
                            rank = utils.get_rank(score, triples[:, 2])
                            rank_list.append(rank.cpu())
                            batch_triples = np.concatenate((batch_triples, triples.cpu().numpy()), axis=0)
                            select_chains = np.concatenate((select_chains, select_chain), axis=0)
                        test_triples.append(batch_triples)
                    unvalid_ratio/=len(test_time)
                    if epoch == 0:
                        test_unknown_triple_index = utils.get_unkown_index(test_triples, unkown_entity_index)
                        test_emerging_triple_index = utils.get_emerging_index(test_triples, known_entity_train_val_index)
                        print(f"Unknown triples in test set: {len(test_unknown_triple_index)}, Emerging triples in test set: {len(test_emerging_triple_index)}")
                    mrr, hit1, hit3, hit10 = utils.get_metric(rank_list)
                    unknown_mrr_test, unknown_hit1, unknown_hit3, unknown_hit10 = utils.get_metric_unknown_both(rank_list, test_triples, test_unknown_triple_index)
                    emerging_mrr_test, emerging_hit1, emerging_hit3, emerging_hit10 = utils.get_metric_emerging_both(rank_list, test_triples, test_emerging_triple_index)

                    print(f"Epoch: {epoch}, Test Loss: {test_loss/len(test_loss_list):.4f}")
                    print(f"MRR: {mrr:.4f}, Hit1: {hit1:.4f}, Hit3: {hit3:.4f}, Hit10: {hit10:.4f}")
                    print(f"Unknown_MRR: {unknown_mrr_test:.4f}, Unknown_Hit1: {unknown_hit1:.4f}, Unknown_Hit3: {unknown_hit3:.4f}, Unknown_Hit10: {unknown_hit10:.4f}")
                    print(f"Emerging_MRR: {emerging_mrr_test:.4f}, Emerging_Hit1: {emerging_hit1:.4f}, Emerging_Hit3: {emerging_hit3:.4f}, Emerging_Hit10: {emerging_hit10:.4f}")
                    best_test_mertic = f"Unknown_MRR: {unknown_mrr_test:.4f}, Unknown_Hit1: {unknown_hit1:.4f}, Unknown_Hit3: {unknown_hit3:.4f}, Unknown_Hit10: {unknown_hit10:.4f}"
                    best_result = {'all_mrr': mrr, 'all_hit1': hit1, 'all_hit3': hit3, 'all_hit10': hit10,
                                   'unknown_mrr': unknown_mrr_test, 'unknown_hit1': unknown_hit1, 'unknown_hit3': unknown_hit3, 'unknown_hit10': unknown_hit10,
                                   'emerging_mrr': emerging_mrr_test, 'emerging_hit1': emerging_hit1, 'emerging_hit3': emerging_hit3, 'emerging_hit10': emerging_hit10}
                patience = 0
                ## save the model
                model_path = os.path.join(results_dir, model_name)
                embedding_path = os.path.join(results_dir, f"embedding_dataset_{args.dataset}_history_len_{args.history_len}_max_length_{args.max_length}_hidden_dim_{args.hidden_dim}_num_layers_{args.num_layers}_num_heads_{args.num_heads}_num_codes_{args.num_code}_tips_{args.tips}_seed_{seed}.pth")
                torch.save(model.state_dict(), model_path)
                torch.save(embedding_dict, embedding_path)
            else:
                patience += 1
            if patience > max_patience:
                print(f"Early stopping at epoch {epoch}, best validation MRR: {best_val_mrr:.4f}")
                print(f"Best test metrics: {best_test_mertic}")
                best_result_dict[seed] = best_result
                break
            if epoch == args.epochs - 1:
                print(f"Reached maximum epochs {args.epochs}, best validation MRR: {best_val_mrr:.4f}")
                print(f"Best test metrics: {best_test_mertic}")
                best_result_dict[seed] = best_result

    
    print("Best results for each seed:")
    for seed, result in best_result_dict.items():
        print(f"Seed {seed}: MRR: {result['all_mrr']:.4f}, Hit1: {result['all_hit1']:.4f}, Hit3: {result['all_hit3']:.4f}, Hit10: {result['all_hit10']:.4f}, "
              f"Unknown_MRR: {result['unknown_mrr']:.4f}, Unknown_Hit1: {result['unknown_hit1']:.4f}, Unknown_Hit3: {result['unknown_hit3']:.4f}, Unknown_Hit10: {result['unknown_hit10']:.4f}"
              f"Emerging_MRR: {result['emerging_mrr']:.4f}, Emerging_Hit1: {result['emerging_hit1']:.4f}, Emerging_Hit3: {result['emerging_hit3']:.4f}, Emerging_Hit10: {result['emerging_hit10']:.4f}")
    print("Average results:")
    avg_result = {}
    for key in result.keys():
        avg_result[key] = np.mean([result[key] for result in best_result_dict.values()])
    print(f"Average MRR: {avg_result['all_mrr']:.4f}, Average Hit1: {avg_result['all_hit1']:.4f}, Average Hit3: {avg_result['all_hit3']:.4f}, Average Hit10: {avg_result['all_hit10']:.4f}, "
          f"Average Unknown_MRR: {avg_result['unknown_mrr']:.4f}, Average Unknown_Hit1: {avg_result['unknown_hit1']:.4f}, Average Unknown_Hit3: {avg_result['unknown_hit3']:.4f}, Average Unknown_Hit10: {avg_result['unknown_hit10']:.4f}"
          f"Average Emerging_MRR: {avg_result['emerging_mrr']:.4f}, Average Emerging_Hit1: {avg_result['emerging_hit1']:.4f}, Average Emerging_Hit3: {avg_result['emerging_hit3']:.4f}, Average Emerging_Hit10: {avg_result['emerging_hit10']:.4f}")



if __name__ == "__main__":
    args = parse_args()
    main(args)