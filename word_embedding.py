from transformers import BertTokenizer, BertModel
import os
import torch
import numpy as np
import argparse


def get_entity_embedding_bert(entity_name, tokenizer, model):
    entity_name = entity_name.replace("_", " ").replace("(", "").replace(")", "")
    
    inputs = tokenizer(entity_name, return_tensors='pt', add_special_tokens=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state
    
    cls_embedding = last_hidden_state[0][0].numpy()  
    
    return cls_embedding


def load_string(file_path):
    with open(file_path, 'r') as fr:
        entity2str = []
        for line in fr:
            line_split = line.split()
            
            length = len(line_split)
            str = ""
            for i in range(length - 1):
                str += line_split[i] + " "
            id = int(line_split[-1])
            entity2str.append(str)
    return entity2str

def get_entity_embedding_bert(entity_name, tokenizer, model):
    entity_name = entity_name.replace("_", " ").replace("(", "").replace(")", "")
    
    inputs = tokenizer(entity_name, return_tensors='pt', add_special_tokens=True)
    inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state
    
    cls_embedding = last_hidden_state[0][0].detach().cpu().numpy()  
    
    return cls_embedding

def main(args):
    bert_model_path = args.bert_model_path
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = BertModel.from_pretrained(bert_model_path).to('cuda')


    dataset = args.dataset
    file_path = f'data/{dataset}/entity2id.txt'
    entity_str = load_string(file_path)
    print(entity_str)
    if dataset == 'GDELT':
        entity_str_list = []
        for entity in entity_str:
            entity_s = entity.split('(')[0]
            entity_str_list.append(entity_s.strip())
            # break
    else:
        entity_str_list = []
        for entity in entity_str:
            entity_s = entity.split('	')[0]
            entity_str_list.append(entity_s.strip())
            # break
        

    entity_embeddings = np.zeros((len(entity_str), 768))
    print(len(entity_str_list))
    for i, entity in enumerate(entity_str_list):
        print(f"Processing entity {i+1}/{len(entity_str_list)}: {entity}")
        entity_embeddings[i] = get_entity_embedding_bert(entity, tokenizer, model)

    np.save(f'data/{dataset}/{dataset}_Bert_Entity_Embedding.npy', entity_embeddings)


def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocess for TransFIR")
    parser.add_argument("--dataset", type=str, default="ICEWS14", help="Dataset name (default: ICEWS14)")
    parser.add_argument("--bert_model_path", type=str, default="your_model", help="Path to pre-trained BERT model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)