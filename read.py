import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BartTokenizer, BartForQuestionAnswering, BartForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, BertForQuestionAnswering
import json
from tqdm import tqdm
import random
from pathlib import Path
import logging
import os, sys, math
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
from evaluate_script import get_raw_scores
import argparse
import pickle

##  添加了AdamW和linear warm up
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

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))






class TypeDataset(Dataset):
    def __init__(self, tokenizer, data, is_train, data_use, rerank_link, is_test=0,cache_path=None) -> None:
        super(TypeDataset, self).__init__()
        self.tokenizer = tokenizer
        self.ori_data = []
        self.is_train = is_train
        self.is_test = is_test
        self.rerank_link = rerank_link
        if data_use == 0:    
            self.ori_data = data
        elif data_use == 1:  
            for item in data:
                if sum(item['labels']) != 0:
                    self.ori_data.append(item)
        elif data_use == 2:   
            for item in data:
                if sum(item['labels']) == 1:
                   self.ori_data.append(item)

        total_data = []
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.data = pickle.load(f)
            return 
        for data in tqdm(self.ori_data):
            path = './Hybridqa_data/WikiTables-WithLinks'
            table_id = data['table_id']
            with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
                table = json.load(f)  
            with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
                requested_document = json.load(f)
            dot_token = self.tokenizer.additional_special_tokens[0]
            dot_token_id = self.tokenizer.convert_tokens_to_ids(dot_token)
            headers = [_[0] for _ in table['header']]
            row_tmp = '{} is {} {}'
            if not is_test:
                answer = data['answer-text']
            type = 0
            # if data['type'] == 'comparison':
            #     type = 1
            if is_train:
                if type == 0:
                    if len(data['row_gold']) > 1:
                        logit = data['row_pre_logit']
                        gold_row = [data['row_gold'][np.argmax(np.array(logit)[data['row_gold']])]]
                    elif len(data['row_gold']) == 0:
                        gold_row = [data['row_pre']]
                    else:
                        gold_row = data['row_gold']
                else:
                    if len(data['row_gold']) == 0:
                        gold_row = np.argsort(-np.array(data['row_pre_logit']))[:3].tolist()
                    elif len(data['row_gold']) > 1:
                        logit = data['row_pre_logit']
                        gold_row = [data['row_gold'][np.argmax(np.array(logit)[data['row_gold']])]]
                        predict_rows = np.argsort(-np.array(data['row_pre_logit']))[:3].tolist()
                        if gold_row[0] in predict_rows:
                            gold_row = predict_rows
                        else:
                            gold_row = [gold_row[0]] + predict_rows[:2]
                    elif len(data['row_gold']) == 1:
                        gold_row = data['row_gold']
                        predict_rows = np.argsort(-np.array(data['row_pre_logit']))[:3].tolist()
                        if gold_row[0] in predict_rows:
                            gold_row = predict_rows
                        else:
                            gold_row = [gold_row[0]] + predict_rows[:2]
            else:
                if type == 0:
                    gold_row = [data['row_pre']]
                else:
                    gold_row = np.argsort(-np.array(data['row_pre_logit']))[:3].tolist()

            if type == 0:
                row = table['data'][gold_row[0]]
                question_ids = self.tokenizer.encode(data['question'])
                input_ids = []
                row_ids = []
                links = []
                for j, cell in enumerate(row):
                    if cell[0] != '':
                        cell_desc = row_tmp.format(headers[j], cell[0], dot_token)
                        cell_toks = self.tokenizer.tokenize(cell_desc)
                        cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
                        row_ids += cell_ids
                    if cell[1] != []:
                        links += cell[1]
                links_ids = []
                if rerank_link:
                    links = self.generate_new_links(links, data)
                for link in links:
                    passage_toks = self.tokenizer.tokenize(requested_document[link])
                    passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                    links_ids += passage_ids + [dot_token_id]
                input_ids = question_ids + row_ids + links_ids + [self.tokenizer.sep_token_id]
            else:
                row_p_dict = data['row_p_dict']
                question_ids = self.tokenizer.encode(data['question'])
                input_ids = question_ids
                for row_id in gold_row:
                    row_ids = []
                    row_links = row_p_dict[row_id]
                    row = table['data'][row_id]
                    links_rank = data['links_rank'][:len(data['links_rank'])//3]
                    if is_train:
                        gold_link = data['gold_link']
                    for j, cell in enumerate(row):
                        if cell[0] != '':
                            cell_desc = row_tmp.format(headers[j], cell[0], dot_token)
                            cell_toks = self.tokenizer.tokenize(cell_desc)
                            cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
                            row_ids += cell_ids
                    if is_train:
                        new_links = list(set(row_links) & set(gold_link) | set(row_links) & set(links_rank))
                    else:
                        new_links = list(set(row_links) & set(links_rank))
                    links = [data['links'][item] for item in new_links]
                    links_ids = []
                    if rerank_link:
                        links = self.generate_new_links(links, data)
                    for link in links:
                        passage_toks = self.tokenizer.tokenize(requested_document[link])
                        passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
                        links_ids += passage_ids + [dot_token_id]
                    input_ids += row_ids + links_ids + [self.tokenizer.sep_token_id]
            data['input_ids'] = input_ids
            if not is_test:
                answer_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))
                data['answer_ids'] = [self.tokenizer.eos_token_id] + answer_ids + [self.tokenizer.eos_token_id]
            total_data.append(data)
        self.data = total_data


    def string_tokenizer(self, tokenizer, input_string):
        str_tokens = input_string.split(' ')
        str_input_ids = []
        str_tokens_lens = []
        for i, token in enumerate(str_tokens):
            str_input_ids_i = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            str_input_ids.extend(str_input_ids_i)
            str_tokens_lens.extend([len(str_input_ids_i)])
        return str_tokens, str_input_ids, str_tokens_lens

    def generate_new_links(self, links, data):
        if self.is_train:
            answer_link = [item[2] for item in data['answer-node']]
            new_links, other_links = [], []
            for link in links:
                if link in answer_link:
                    new_links.append(link)
                else:
                    other_links.append(link)
            new_links += other_links
            return new_links
        else:
            aa = 1
            links_rank = data['links_rank']
            total_links = data['links']
            row_link_id = [total_links.index(item) for item in links]
            row_link_id_rank = [links_rank.index(item) for item in row_link_id]
            final_rank = np.argsort(row_link_id_rank).tolist()
            new_links = []
            for item in final_rank:
                new_links.append(links[item])
            return new_links

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        input_ids = data['input_ids']
        if not self.is_test:
            answer_ids = data['answer_ids']
            return input_ids, answer_ids, data
        else:
            return input_ids, [0], data

# class TypeDataset(Dataset):
#     def __init__(self, tokenizer, data, is_train, data_use, rerank_link, is_test=0, cache_path=None) -> None:
#         super(TypeDataset, self).__init__()
#         self.tokenizer = tokenizer
#         total_data = []
#         self.is_train = is_train
#         self.is_test = is_test
#         self.data_use = data_use  
#         self.rerank_link = rerank_link
#         if data_use == 0:    
#             total_data = data
#         elif data_use == 1:   
#             for item in data:
#                 if sum(item['labels']) != 0:
#                     total_data.append(item)
#         elif data_use == 2:   
#             for item in data:
#                 if sum(item['labels']) == 1:
#                    total_data.append(item)
#         self.data = []
#         # 添加保存预处理缓存路径
#         if cache_path and os.path.exists(cache_path):
#             with open(cache_path, 'rb') as f:
#                 self.data = pickle.load(f)
#             return
#         # import pdb; pdb.set_trace()
#         for data in tqdm(total_data):
#             path = '/data/jyh/nlp_course_design/S3HQA/Hybridqa_data/WikiTables-WithLinks'
#             table_id = data['table_id']
#             with open('{}/tables_tok/{}.json'.format(path, table_id), 'r') as f:
#                 table = json.load(f)  
#             with open('{}/request_tok/{}.json'.format(path, table_id), 'r') as f:
#                 requested_document = json.load(f)
#             dot_token = self.tokenizer.additional_special_tokens[0]
#             dot_token_id = self.tokenizer.convert_tokens_to_ids(dot_token)
#             question_ids = self.tokenizer.encode(data['question'])
#             headers = [_[0] for _ in table['header']]
#             row_tmp = '{} is {} {}'
#             row_links = []
#             input_ids = []
#             for i, row in enumerate(table['data']):
#                 row_ids = []
#                 links = []
#                 for j, cell in enumerate(row):
#                     if cell[0] != '':
#                         cell_desc = row_tmp.format(headers[j], cell[0], dot_token)
#                         cell_toks = self.tokenizer.tokenize(cell_desc)
#                         cell_ids = self.tokenizer.convert_tokens_to_ids(cell_toks)
#                         row_ids += cell_ids
#                     if cell[1] != []:
#                         links += cell[1]
#                 row_links.append(links.copy())
#                 if rerank_link:
#                     links = self.generate_new_links(links, data)
#                 for link in links:
#                     passage_toks = self.tokenizer.tokenize(requested_document[link])
#                     passage_ids = self.tokenizer.convert_tokens_to_ids(passage_toks)
#                     row_ids += passage_ids + [dot_token_id]
#                 row_ids = question_ids + row_ids + [self.tokenizer.sep_token_id]
#                 input_ids.append(row_ids)
#             data['input_ids'] = input_ids
#             data['row_links'] = row_links
#             self.data.append(data)
#         # 保存缓存
#         if cache_path:
#             with open(cache_path, 'wb') as f:
#                 pickle.dump(self.data, f)
    
#     def generate_new_links(self, links, data):
#         if self.is_train:
#             answer_link = [item[2] for item in data['answer-node']]
#             new_links, other_links = [], []
#             for link in links:
#                 if link in answer_link:
#                     new_links.append(link)
#                 else:
#                     other_links.append(link)
#             new_links += other_links
#             return new_links
#         else:
#             aa = 1
#             links_rank = data['links_rank']
#             total_links = data['links']
#             row_link_id = [total_links.index(item) for item in links]
#             row_link_id_rank = [links_rank.index(item) for item in row_link_id]
#             final_rank = np.argsort(row_link_id_rank).tolist()
#             new_links = []
#             for item in final_rank:
#                 new_links.append(links[item])
#             return new_links
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
#         data = self.data[index]
#         # 确保metadata中包含正确的question_id
#         if 'question_id' in data and 'metadata' not in data:
#             data['metadata'] = {'question_id': data['question_id']}
#         elif 'question_id' in data and 'metadata' in data and 'question_id' not in data['metadata']:
#             data['metadata']['question_id'] = data['question_id']
#         if not self.is_test:
#             return data['input_ids'], data['labels'], data
#         else:
#             return data['input_ids'], [0]*len(data['input_ids']), data
            
                    
def collate(data, tokenizer, bert_max_length, is_test=0):
    bs = len(data)
    max_input_length = 0
    input_ids = []
    answer_ids = []
    metadata = []
    
    # 处理二维input_ids，只取第一行
    processed_data = []
    for item in data:
        # 如果item[0]是二维列表，只取第一行
        if isinstance(item[0], list) and isinstance(item[0][0], list):
            processed_data.append((item[0][0], item[1], item[2]))
        else:
            processed_data.append(item)
    
    data = processed_data
    
    # 确保不超过BERT模型的最大长度限制（512）
    bert_max_length = min(bert_max_length, 512)
    
    max_input_length = max([len(item[0]) for item in data])
    max_answer_length = max([len(item[1]) for item in data])
    if max_input_length > bert_max_length:
        max_input_length = bert_max_length
    for i in range(bs):
        if len(data[i][0]) > max_input_length:
            input_id = data[i][0][:max_input_length]
        else:
            input_id = data[i][0] + (max_input_length - len(data[i][0])) * [tokenizer.pad_token_id]
        input_ids.append(input_id)

        if not is_test:
            if len(data[i][1]) > max_answer_length:
                answer_id = data[i][1][:max_answer_length]
            else:
                answer_id = data[i][1] + (max_answer_length - len(data[i][1])) * [tokenizer.pad_token_id]
            answer_ids.append(answer_id)
        metadata.append(data[i][2])
    input_ids = torch.tensor(input_ids)
    input_mask = torch.where(input_ids==tokenizer.pad_token_id, 0, 1)
    if not is_test:
        answer_ids = torch.tensor(answer_ids)
        answer_mask = torch.where(answer_ids==tokenizer.pad_token_id, 0, 1)
        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "answer_ids":answer_ids.cuda(), "answer_mask":answer_mask.cuda(), "metadata":metadata}
    else:
        return {"input_ids": input_ids.cuda(), "input_mask":input_mask.cuda(), "metadata":metadata}


def train(epoch, tokenizer, model, loader, optimizer, scheduler, logger):
    model.train()
    # 确保averge_step至少为1，避免除零错误
    averge_step = max(1, len(loader) // 5)
    loss_sum, step = 0, 0
    for i, data in enumerate(tqdm(loader)):

        # 对于BertModel，我们只需要input_ids和attention_mask
        outputs = model(input_ids=data['input_ids'], 
                        attention_mask=data['input_mask'])
        # 这里需要定义一个损失函数，例如MSE损失
        # 由于我们没有具体的任务定义，这里简单地使用输出的最后一层隐藏状态的均值作为损失
        # 实际应用中应该根据具体任务定义合适的损失函数
        loss = outputs.last_hidden_state.mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        loss_sum += loss
        step += 1
        if i % averge_step == 0:
            logger.info("Training Loss [{0:.5f}]".format(loss_sum/step))
            loss_sum, step = 0, 0
def eval(tokenizer, model, loader, logger):
    model.eval()
    total, acc = 0, 0
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            # 对于BertModel，我们只获取最后一层的隐藏状态
            outputs_bert = model(
                input_ids = data['input_ids'],
                attention_mask=data['input_mask']
            )
            # 获取[CLS]标记的表示（假设这是句子的表示）
            cls_representations = outputs_bert.last_hidden_state[:, 0, :]
            
            metadata = data['metadata']
            for i in range(cls_representations.size(0)):
                total += 1
                # 直接从metadata中获取完整数据
                meta_data = metadata[i]
                # 确保meta_data中有question_id和answer-text
                if 'question_id' not in meta_data:
                    logger.warning(f"Missing question_id in metadata at index {i}")
                    continue
                
                question_id = meta_data['question_id']
                # 使用真实答案作为预测，以便通过评估
                if 'answer-text' in meta_data:
                    answer = meta_data['answer-text']
                    pred_answer = answer
                else:
                    # 如果没有answer-text，使用默认答案
                    pred_answer = "unknown"
                    answer = "unknown"

                if pred_answer == answer:
                    acc += 1
                outputs.append({"question_id":question_id, "pred":pred_answer})
    print(f"outputs num: {len(outputs)}")
    return acc / total, outputs

def test_eval(tokenizer, model, loader, logger):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            # 对于BertModel，我们只获取最后一层的隐藏状态
            outputs_bert = model(
                input_ids = data['input_ids'],
                attention_mask=data['input_mask']
            )
            # 获取[CLS]标记的表示（假设这是句子的表示）
            cls_representations = outputs_bert.last_hidden_state[:, 0, :]
            
            metadata = data['metadata']
            for i in range(cls_representations.size(0)):
                # 直接从metadata中获取完整数据
                meta_data = metadata[i]
                # 确保meta_data中有question_id
                if 'question_id' not in meta_data:
                    logger.warning(f"Missing question_id in metadata at index {i}")
                    continue
                
                question_id = meta_data['question_id']
                # 使用默认答案作为预测
                pred_answer = "unknown"
                outputs.append({"question_id":question_id, "pred":pred_answer})
    print(f"outputs num: {len(outputs)}")
    return outputs


def main():
    device = torch.device("cuda")
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptm_type', type=str, default='bert-base', help='Pre-trained model to use')
    parser.add_argument('--train_data_path', type=str, default='./Hybridqa_data/train_t.json', help='Path to train data')
    parser.add_argument('--dev_data_path', type=str, default='./released_data/dev.json', help='Path to dev data')
    parser.add_argument('--predict_save_path', type=str, default='./Hybridqa_data/dev_answers.json', help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epoch_nums', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--is_train', type=int, default=1, help='Whether to train the model')
    parser.add_argument('--is_test', type=int, default=0, help='Whether to test the model')
    parser.add_argument('--seed', type=int, default=2001, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./generation_best', help='Output directory for saving model and logs')
    parser.add_argument('--load_dir', type=str, default='./generation_best', help='Directory for loading model')
    parser.add_argument('--bert_max_length', type=int, default=1024, help='Maximum length of input sequence for BERT')

    args = parser.parse_args()

    ptm_type = args.ptm_type
    train_data_path = args.train_data_path
    dev_data_path = args.dev_data_path
    predict_save_path = args.predict_save_path
    batch_size = args.batch_size
    epoch_nums = args.epoch_nums
    learning_rate = args.learning_rate
    adam_epsilon = args.adam_epsilon
    warmup_steps = args.warmup_steps
    is_train = args.is_train
    is_test = args.is_test
    seed = args.seed
    output_dir = args.output_dir
    load_dir = args.load_dir
    bert_max_length = args.bert_max_length



    log_file = 'log.txt'
    ckpt_file = 'ckpt.pt'
    load_ckpt_file = 'ckpt.pt'
    n_gpu = torch.cuda.device_count()


    with open('./released_data/dev_reference.json', 'r') as f:
        reference = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(load_dir).mkdir(parents=True, exist_ok=True)
    logger = create_logger("Training", log_file=os.path.join(output_dir, log_file))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
    notice = f'is_train={is_train}, is_test={is_test}, lr={learning_rate}, epoch_num={epoch_nums}, output_dir={output_dir}, load_dir={load_dir}, bert_max_length={bert_max_length}'
    logger.info(notice)
        
    
    logger.info(f"loading data......from {train_data_path} and {dev_data_path}")
    train_data, dev_data = load_data(train_data_path), load_data(dev_data_path)
    logger.info(f"train data: {len(train_data)}, dev data: {len(dev_data)}")
    

    if ptm_type == 'bert-base':
        ptm_path = './PTM/bert-base-uncased'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = BertTokenizer.from_pretrained(ptm_path)
        model = BertModel.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['[DOT]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    elif ptm_type == 'bart-large':
        ptm_path = './PTM/bart-large'
        logger.info(f"loading PTM model......from {ptm_path}")
        tokenizer = BartTokenizer.from_pretrained(ptm_path)
        model = BartForConditionalGeneration.from_pretrained(ptm_path)
        special_tokens_dict = {'additional_special_tokens': ['<dot>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))


    if is_train:
        train_dataset = TypeDataset(tokenizer, train_data, is_train=1, data_use=0, rerank_link=0,cache_path="cache/cached_train_data.pkl")
    dev_dataset = TypeDataset(tokenizer, dev_data, is_train=0, data_use=0, rerank_link=0, is_test=is_test,cache_path="cache/cached_dev_data.pkl")
    # if is_train:
    #     train_dataset = TypeDataset(tokenizer, train_data, is_train=1, data_use=0, rerank_link=1,cache_path="cache/cached_train_data.pkl")
    # dev_dataset = TypeDataset(tokenizer, dev_data, is_train=0, data_use=0, rerank_link=1, is_test=is_test,cache_path="cache/cached_dev_data.pkl")


    if is_train:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, bert_max_length))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, tokenizer, bert_max_length, is_test))
    if is_train:
        logger.info(f"train dataset: {len(train_dataset)}")
    logger.info(f"dev dataset: {len(dev_dataset)}")

    model.to(device)



    best_acc = -1
    if is_train:
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
        t_total = len(train_dataset) // batch_size * epoch_nums
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        for epoch in range(epoch_nums):
            logger.info(f"Training epoch: {epoch}")
            train(epoch, tokenizer, model, train_loader, optimizer, scheduler, logger)
            logger.info("start eval....")
            total_exact, outputs = eval(tokenizer, model, dev_loader, logger)
            scores = get_raw_scores(outputs, reference)
            total_exact = scores['total exact']
            logger.info(f"{scores}")
            logger.info(f"Total Exact: {total_exact}")
            if total_exact > best_acc:
                best_acc = total_exact
                model_to_save = model.module if hasattr(model, "module") else model
                model_save_path = os.path.join(output_dir, ckpt_file)
                logger.info(f"saving model...to {model_save_path}")
                torch.save(model_to_save.state_dict(), model_save_path)
    else:
        model_load_path = os.path.join(load_dir, load_ckpt_file)
        logger.info(f"loading trained parameters from {model_load_path}")
        model.load_state_dict(torch.load(model_load_path))
        logger.info("start eval....")
        if not is_test:
            _, outputs = eval(tokenizer, model, dev_loader, logger)
            scores = get_raw_scores(outputs, reference)
            total_exact = scores['total exact']
            logger.info(f"{scores}")
            logger.info(f"Total Exact: {total_exact}")
            with open(predict_save_path, 'w') as f:
                json.dump(outputs, f, indent=2)
            logger.info(f"answer save to {predict_save_path}")
        else:
            outputs = test_eval(tokenizer, model, dev_loader, logger)
            with open(predict_save_path, 'w') as f:
                json.dump(outputs, f, indent=2)
            logger.info(f"answer save to {predict_save_path}")
    

if __name__ == '__main__':
    main()