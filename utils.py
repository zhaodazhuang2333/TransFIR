import numpy as np
import torch
from tqdm import tqdm
import knowledge_graph as knwlgrh
from collections import defaultdict
import pdb


def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI"]:
        return knwlgrh.load_from_local("./data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans
    

def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    return all_ans_list

def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def get_rank(score, label):
    # pdb.set_trace()
    _, indices = torch.sort(score, dim=1, descending=True)
    label_indices = label.unsqueeze(1)
    ranks = (indices == label_indices).nonzero(as_tuple=False)[:, 1] + 1
    return ranks

def get_rank_filter(score, triples):
    """
    score: Tensor of shape (T, num_entities), predicted scores for each triple (predicting tail)
    triples: Tensor of shape (T, 3), where each row is (s, r, o) at current timestamp
    Returns: Tensor of shape (T,), the filtered rank of each true o
    """
    T, num_entities = score.shape
    triples = triples.tolist()
    all_triplets_set = set((s, r, o) for s, r, o, t in triples)  # 当前时间点下的所有真实 triple
    ranks = []
    for i in range(T):
        s, r, o, t = triples[i]

        # 过滤掉除当前 o 以外的所有 (s, r, o') 正确三元组
        filter_mask = torch.ones(num_entities, dtype=torch.bool, device=score.device)
        for candidate_o in range(num_entities):
            if candidate_o != o and (s, r, candidate_o) in all_triplets_set:
                filter_mask[candidate_o] = False

        filtered_score = score[i].clone()
        filtered_score[~filter_mask] = -float("inf")

        # 获取排名
        _, indices = torch.sort(filtered_score, descending=True)
        rank = (indices == o).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)
    return torch.tensor(ranks, device=score.device)



def get_metric(rank):
    rank = torch.cat(rank)
    mrr = torch.mean(1.0 / rank.float())
    hits1 = torch.mean((rank <= 1).float())
    hits3 = torch.mean((rank <= 3).float())
    hits10 = torch.mean((rank <= 10).float())
    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

def get_unkown_index(valid_triples, unknown_entity_index):
    valid_triples = np.concatenate(valid_triples, axis=0)
    in_triple = np.isin(valid_triples[:, [0, 2]], list(unknown_entity_index))
    unknown_index = np.where(np.any(in_triple, axis=1))[0]
    return torch.tensor(unknown_index, dtype=torch.long)


def get_metric_unknown_both(rank, valid_triples, unknown_index):
    rank = torch.cat(rank)
    valid_triples = np.concatenate(valid_triples, axis=0)
    count = 0
    rank = rank[unknown_index]
    valid_triples = valid_triples[unknown_index]
    mrr = torch.mean(1.0 / rank.float())
    hits1 = torch.mean((rank <= 1).float())
    hits3 = torch.mean((rank <= 3).float())
    hits10 = torch.mean((rank <= 10).float())
    return mrr.item(), hits1.item(), hits3.item(), hits10.item()


def get_emerging_index(triples, known_entity_index):
    known_entity_index = set(
    int(e) if isinstance(e, torch.Tensor) else e
    for e in known_entity_index
)

    emerging_triple_indices = []
    global_index = 0

    for triple_batch in triples:
        h = triple_batch[:, 0]
        t = triple_batch[:, 2]
        batch_emerging_indices = []
        new_entities = set()

        for i in range(triple_batch.shape[0]):
            h_id = h[i].item()
            t_id = t[i].item()

            # 如果 h 或 t 是新的，就标记该三元组为 emerging
            if h_id not in known_entity_index or t_id not in known_entity_index:
                batch_emerging_indices.append(global_index)

            # 把当前时间步中所有出现的实体暂存在 new_entities
            new_entities.update([h_id, t_id])
            global_index += 1

        # 时间步结束后统一更新已知实体集合
        known_entity_index.update(new_entities)
        emerging_triple_indices.extend(batch_emerging_indices)

    return torch.tensor(emerging_triple_indices, dtype=torch.long)




def get_metric_emerging_both(rank, valid_triples, emerging_index):
    rank = torch.cat(rank)
    valid_triples = np.concatenate(valid_triples, axis=0)
    rank = rank[emerging_index]
    mrr = torch.mean(1.0 / rank.float())
    hits1 = torch.mean((rank <= 1).float())
    hits3 = torch.mean((rank <= 3).float())
    hits10 = torch.mean((rank <= 10).float())
    return mrr.item(), hits1.item(), hits3.item(), hits10.item()



def get_metric_unknown(rank, inv_rank, valid_triples, unknown_entity_index):
    rank = torch.cat(rank)
    inv_rank = torch.cat(inv_rank)
    # pdb.set_trace()
    valid_triples = np.concatenate(valid_triples, axis=0)
    in_triple = np.isin(valid_triples[:, [0, 2]], list(unknown_entity_index))
    unknown_index = np.where(np.any(in_triple, axis=1))[0]
    rank = rank[unknown_index]
    inv_rank = inv_rank[unknown_index]
    # pdb.set_trace()
    mrr = torch.mean(1.0 / rank.float())
    hits1 = torch.mean((rank <= 1).float())
    hits3 = torch.mean((rank <= 3).float())
    hits10 = torch.mean((rank <= 10).float())
    mrr_inv = torch.mean(1.0 / inv_rank.float())
    hits1_inv = torch.mean((inv_rank <= 1).float())
    hits3_inv = torch.mean((inv_rank <= 3).float())
    hits10_inv = torch.mean((inv_rank <= 10).float())
    avg_mrr = (mrr + mrr_inv) / 2
    avg_hits1 = (hits1 + hits1_inv) / 2
    avg_hits3 = (hits3 + hits3_inv) / 2
    avg_hits10 = (hits10 + hits10_inv) / 2
    return avg_mrr.item(), avg_hits1.item(), avg_hits3.item(), avg_hits10.item()