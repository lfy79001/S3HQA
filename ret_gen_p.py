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



class TypeDataset(Dataset):
    def __init__(self, tokenizer, input_data, is_train, MIL, JT) -> None:
        super(TypeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.is_train = is_train
        self.MIL = MIL  
        self.JT = JT   
        
        for i, data in enumerate(tqdm(input_data)):
            if sum(data['labels']) != 0:
                continue
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
                label = 0
                # if data['answer-text'] in requested_document[link]:
                #     label = 1
                labels.append(label)
            data['links'] = links
            data['link_labels'] = labels
            data['input_ids'] = input_ids
            self.data.append(data)
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        return data['input_ids'], data
            
                    
def collate(data, tokenizer, bert_max_length, is_train):
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


def train(epoch, tokenizer, model, loader, optimizer, logger):
    model.train()
    averge_step = len(loader) // 3
    loss_sum, step = 0, 0
    for i, data in enumerate(tqdm(loader)):
        probs = model(data)
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        loss = loss_func(probs, data['link_labels'])
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
    total, acc = 0, 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            predicts = torch.argmax(F.softmax(probs,dim=1), dim=1).cpu().tolist()
            labels = data['link_labels'].cpu().tolist()
            for i in range(len(predicts)):
                total += 1
                if predicts[i] == labels[i]:
                    acc += 1
    return acc / total

def eval_dev(model, loader, logger):
    model.eval()
    total, acc1, acc2, acc3, acc5 = 0, 0, 0, 0, 0
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            probs = model(data)
            metadata = data['metadata']
            link_rank = torch.argsort(F.softmax(probs,dim=1)[:,1], descending=True).cpu().detach().tolist()
            labels = metadata['link_labels']
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
    dev_data_path = '/home/lfy/UMQM/Data/HybridQA/train2.json'
    predict_save_path = '/home/lfy/UMQM/Data/HybridQA/train.toy.json'
    
    batch_size = 1
    epoch_nums = 10
    learning_rate = 5e-5
    is_train = 0
    seed = 2001
    output_dir = './retrievep_deberta1'
    load_dir = './retrievep_deberta1'  
    log_file = 'logp.txt'
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
        
    dev_dataset = TypeDataset(tokenizer, dev_data, is_train, MIL, JT)
    logger.info(f"dev dataset: {len(dev_dataset)}")
    dev_loader = DataLoader(dev_dataset, batch_size=1, collate_fn=lambda x: collate(x, tokenizer, bert_max_length, is_train))
    
    model = RetrieveModel(bert_model)
    model.to(device)

    best_acc = 0
    if is_train:
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        total_loss, step = 0, 0
        for epoch in range(epoch_nums):
            logger.info(f"Training epoch: {epoch}")
            train(epoch, tokenizer, model, train_loader, optimizer, logger)
            logger.info("start eval....")
            acc = eval(model, dev_loader, logger)
            logger.info(f"acc... {acc}")
            if acc > best_acc:
                best_acc = acc
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(output_dir, ckpt_file)
                logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)
    else:
        model_load_path = os.path.join(load_dir, load_ckpt_file)
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start eval....")
        outputs = eval_dev(model, dev_loader, logger)
        with open(predict_save_path, 'w') as f:
            json.dump(outputs, f, indent=2)
        logger.info(f"saving to {predict_save_path}")
    

if __name__ == '__main__':
    main()