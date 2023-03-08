import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import random
from pathlib import Path
import logging
import os, sys, math
import torch.nn.functional as F

def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)    

class RetrieveModel(nn.Module):
    def __init__(self, bert_model):
        super(RetrieveModel, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = self.bert_model.embeddings.word_embeddings.embedding_dim
        self.projection = FFNLayer(self.hidden_size, self.hidden_size, 2, 0.2)

    def forward(self, data):
        inputs = {"input_ids": data['input_ids'], "attention_mask": data['input_mask']}
        cls_output = self.bert_model(**inputs)[0][:,0,:]
        logits = self.projection(cls_output)
        return logits


class TypeDevDataset(Dataset):
    def __init__(self, tokenizer, input_data, is_train, MIL, JT, is_test) -> None:
        super(TypeDevDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = input_data
        self.is_train = is_train
        self.MIL = MIL  # 用多少数据进行训练，用全部还是sum=1
        self.JT = JT   # 为0是普通的row+passage 1是单独训练
        self.is_test = is_test
        
        for i, data in enumerate(tqdm(self.data)):
            table_id = data['table_id']
            input_ids = []
            labels = []
            path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
            with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
                table = json.load(f)  
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)
            links = []
            for i, row in enumerate(table['data']):
                for j, cell in enumerate(row):
                    if cell[1] != []:
                        links += cell[1]
            for i, link in enumerate(links):
                question_ids = self.tokenizer.encode(data['question'])
                passage_toks = self.tokenizer.tokenize(requested_document[link])
                passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                input_id = question_ids + passage_ids + [self.tokenizer.sep_token_id]
                input_ids.append(input_id)
                if not is_test:
                    label = 0
                    if data['answer-text'] in requested_document[link]:
                        label = 1
                    labels.append(label)
            data['links'] = links
            if not is_test:
                data['labels'] = labels
            data['input_ids'] = input_ids
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        return data['input_ids'], data

def dev_collate(data, tokenizer, bert_max_length):
    input_data, metadata = data[0]
    input_ids = []
    max_input_length = max([len(item) for item in input_data])
    if max_input_length > bert_max_length:
        max_input_length = bert_max_length
    for i in range(len(input_data)):
        if len(input_data[i]) > max_input_length:
            input_id = input_data[i][:max_input_length]
        else:
            input_id = input_data[i] + (max_input_length - len(input_data[i])) * [tokenizer.pad_token_id]
        input_ids.append(input_id)
    input_ids = torch.tensor(input_ids)
    input_mask = torch.where(input_ids==tokenizer.pad_token_id, 0, 1)
    return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "metadata":metadata}


class TypeDataset(Dataset):
    def __init__(self, tokenizer, input_data, is_train, MIL, JT) -> None:
        super(TypeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.total_data = []
        self.is_train = is_train
        self.MIL = MIL  # 用多少数据进行训练，用全部还是sum=1
        self.JT = JT   # 为0是普通的row+passage 1是单独训练
            # 把caprison和 sum > 1 的都删掉
        for item in input_data:
            if sum(item['labels']) != 1:
                continue
            self.total_data.append(item)
        if is_train == 1:
            self.data = []
            for data in tqdm(self.total_data):
                path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
                table_id = data['table_id']
                with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
                    table = json.load(f)  
                with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                    requested_document = json.load(f)
                links = []
                answer = data['answer-text']
                answer_links = [item[2] for item in data['answer-node']]
                other_links = []
                for i, row in enumerate(table['data']):
                    for j, cell in enumerate(row):
                        if cell[1] != []:
                            links += cell[1]
                other_links = [link for link in links if link not in answer_links]
                new_links = []
                new_links += answer_links
                shuzhi = min(len(other_links), 5)
                sample_links = random.sample(other_links, shuzhi)
                new_links += sample_links
                for i, link in enumerate(new_links):
                    if not link:
                        continue
                    self.data.append((data['question'], requested_document[link], answer))
        else:
            self.data = self.total_data
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.is_train == 1:
            data = self.data[index]
            question_ids = self.tokenizer.encode(data[0])
            passage_toks = self.tokenizer.tokenize(data[1])
            passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
            input_ids = question_ids + passage_ids + [self.tokenizer.sep_token_id]
            label = 0
        
            if data[2] in data[1]:
                label = 1
            return input_ids, label
        else:
            input_ids = []
            labels = []
            data = self.data[index]
            path = '/home/lfy/UMQM/Data/HybridQA/WikiTables-WithLinks'
            table_id = data['table_id']
            with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
                table = json.load(f)  
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)
            links = []
            answer = data['answer-text']
            for i, row in enumerate(table['data']):
                for j, cell in enumerate(row):
                    if cell[1] != []:
                        links += cell[1]
            for i, link in enumerate(links):
                question_ids = self.tokenizer.encode(data['question'])
                passage_toks = self.tokenizer.tokenize(requested_document[link])
                passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                input_id = question_ids + passage_ids + [self.tokenizer.sep_token_id]
                input_ids.append(input_id)
                label = 0
                if data['answer-text'] in requested_document[link]:
                    label = 1
                labels.append(label)
            return input_ids, labels
                    
            
                    
def collate(data, tokenizer, bert_max_length):
    bs = len(data)
    max_input_length = 0
    input_ids = []
    labels = []
    max_input_length = max([len(item[0]) for item in data])
    if max_input_length > bert_max_length:
        max_input_length = bert_max_length
    for i in range(bs):
        if len(data[i][0]) > max_input_length:
            input_id = data[i][0][:max_input_length]
        else:
            input_id = data[i][0] + (max_input_length - len(data[i][0])) * [tokenizer.pad_token_id]
        input_ids.append(input_id)
        labels.append(data[i][1])

    input_ids = torch.tensor(input_ids)
    input_mask = torch.where(input_ids==tokenizer.pad_token_id, 0, 1)
    labels = torch.tensor(labels)
    return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "labels":labels.cuda()}


def train(epoch, tokenizer, model, loader, optimizer, logger):
    model.train()
    averge_step = len(loader) // 8
    loss_sum, step = 0, 0
    for i, data in enumerate(tqdm(loader)):
        probs = model(data)
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        loss = loss_func(probs, data['labels'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        step += 1
        if i % averge_step == 0:
            logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
            loss_sum, step = 0, 0

def eval(model, loader, logger):
    model.eval()
    total, acc1, acc2, acc3, acc5 = 0, 0, 0, 0, 0
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            metadata = data['metadata']
            link_rank = torch.argsort(F.softmax(probs,dim=1)[:,1], descending=True).cpu().detach().tolist()
            labels = metadata['labels']
            metadata['links_rank'] = link_rank
            gold_label = np.where(np.array(labels)==1)[0].tolist()
            total += 1
            output_metadata = metadata.copy()
            output_metadata.pop('input_ids')
            outputs.append(output_metadata)
            if set(link_rank[:1]).intersection(set(gold_label)) != set():
                acc1 += 1
            if set(link_rank[:2]).intersection(set(gold_label)) != set():
                acc2 += 1
            if set(link_rank[:3]).intersection(set(gold_label)) != set():
                acc3 += 1
            if set(link_rank[:5]).intersection(set(gold_label)) != set():
                acc5 += 1
    logger.info(f'acc1: {acc1/total}, acc2: {acc2/total}, acc3: {acc3/total}, acc5: {acc5/total}')
    return acc1/total, acc2/total, acc3/total, acc5/total

def eval_dev(model, loader, logger):
    model.eval()
    total, acc1, acc2, acc3, acc5 = 0, 0, 0, 0, 0
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            metadata = data['metadata']
            link_rank = torch.argsort(F.softmax(probs,dim=1)[:,1], descending=True).cpu().detach().tolist()
            labels = metadata['labels']
            metadata['links_rank'] = link_rank
            gold_label = np.where(np.array(labels)==1)[0].tolist()
            total += 1
            metadata.pop('input_ids')
            outputs.append(metadata)
            if set(link_rank[:1]).intersection(set(gold_label)) != set():
                acc1 += 1
            if set(link_rank[:2]).intersection(set(gold_label)) != set():
                acc2 += 1
            if set(link_rank[:3]).intersection(set(gold_label)) != set():
                acc3 += 1
            if set(link_rank[:5]).intersection(set(gold_label)) != set():
                acc5 += 1
    logger.info(f'acc1: {acc1/total}, acc2: {acc2/total}, acc3: {acc3/total}, acc5: {acc5/total}')
    return outputs

def main():
    device = torch.device("cuda")
    ptm_type = 'deberta'
    train_data_path = '/home/lfy/UMQM/Data/HybridQA/train2.json'
    # dev_data_path = '/home/lfy/UMQM/Data/HotpotQA/dataset/dev.qa.json'
    dev_data_path = '/home/lfy/UMQM/Data/HybridQA/dev2.json'
    predict_save_path = '/home/lfy/UMQM/Data/HybridQA/test.p.json'
    
    batch_size = 8
    epoch_nums = 12
    learning_rate = 5e-6
    is_train = 1
    is_test = 0
    seed = 2001
    output_dir = './retrievep_deberta2'
    load_dir = './retrievep_deberta2'  #  1和2就是调整了JT的方式
    log_file = 'log.txt'
    ckpt_file = 'ckpt.pt'
    load_ckpt_file = 'ckpt.pt'
    dataset_name = 'hybridqa'   # hotpotqa/hybridqa
    n_gpu = torch.cuda.device_count()
    MIL = 0
    JT = 0
    bert_max_length = 512

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(load_dir).mkdir(parents=True, exist_ok=True)
    logger = create_logger("Training", log_file=os.path.join(output_dir, log_file))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        

        
    
    logger.info(f"loading data......from {train_data_path} and {dev_data_path}")
    train_data, dev_data = load_data(train_data_path), load_data(dev_data_path)
    logger.info(f"train data: {len(train_data)}, dev data: {len(dev_data)}")
    
    if ptm_type == 'bert-large':
        ptm_path = '/home/lfy/PTM/bert-large-uncased'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = BertTokenizer.from_pretrained(ptm_path)
        bert_model = BertModel.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['[DOT]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        bert_model.resize_token_embeddings(len(tokenizer))
    elif ptm_type == 'bert-base':
        ptm_path = '/home/lfy/PTM/bert-base-uncased'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = BertTokenizer.from_pretrained(ptm_path)
        bert_model = BertModel.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['[DOT]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        bert_model.resize_token_embeddings(len(tokenizer))
    elif ptm_type == 'roberta':
        ptm_path = '/home/lfy/PTM/roberta-large'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = RobertaTokenizer.from_pretrained(ptm_path)
        bert_model = RobertaModel.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['<dot>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        bert_model.resize_token_embeddings(len(tokenizer))
    elif ptm_type == 'deberta':
        ptm_path = '/home/lfy/PTM/deberta-base'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = AutoTokenizer.from_pretrained(ptm_path)
        bert_model = AutoModel.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['[DOT]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        bert_model.resize_token_embeddings(len(tokenizer))
        
    
    if is_train:
        train_dataset = TypeDataset(tokenizer, train_data, 1, MIL, JT)
    dev_dataset = TypeDevDataset(tokenizer, dev_data, 0, MIL, JT, is_test)
    if is_train:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, bert_max_length))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=lambda x: dev_collate(x, tokenizer, bert_max_length))
    if is_train:
        logger.info(f"train dataset: {len(train_dataset)}")
    logger.info(f"dev dataset: {len(dev_dataset)}")
    
    
    
    model = RetrieveModel(bert_model)
    model.to(device)

    best_acc1 = 0
    if is_train:
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        total_loss, step = 0, 0
        for epoch in range(epoch_nums):
            logger.info(f"Training epoch: {epoch}")
            train(epoch, tokenizer, model, train_loader, optimizer, logger)
            logger.info("start eval....")
            acc1, acc2, acc3, acc5 = eval(model, dev_loader, logger)
            if acc1 > best_acc1:
                best_acc1 = acc1
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(output_dir, ckpt_file)
                logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)
    else:
        model_load_path = os.path.join(load_dir, load_ckpt_file)
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start eval....")
        acc, outputs = eval_dev(model, dev_loader, logger)
        logger.info(f"acc... {acc}")
        with open(predict_save_path, 'w') as f:
            json.dump(outputs, f, indent=2)
        logger.info(f"saving to {predict_save_path}")
    

if __name__ == '__main__':
    main()