# S3HQA

This is the project containing source code for the paper [S$^3$HQA: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering](https://arxiv.org/abs/2305.11725) in ACL 2023__. 

## Requirements
```bash
python==3.7
torch==1.7.1+cu110
transformers==4.21.1
```

## Data prepare
Download all data from [hear](https://drive.google.com/file/d/1aVoBWvAE2BBaO5a27xHpgOqKGWzUV0K5/view?usp=sharing) . 

Then `unzip Data.zip` .

Download `bert-base-uncased model`, `deberta-base model`, `bart-large model` from huggingfacehub. Or you can use them directly without downloading the model by changing the code. 

Put `bert-base-uncased model` in `./PTM/bert-base-uncased` and `bart-large model` in  `./PTM/bart-large`.

## Use our retrieval data for your work （such as **LLM**)

If your work just focuses on the reader rather than retrieval.

Directly use `train.row.json`, `dev.row.json` and `test.row.json` for your experiments.




## Training

### Use checkpoint


If you want to get final answers of dev or test set. 

First, download reader checkpoint from [hear](https://drive.google.com/file/d/1IWHY-_kLNyHKZBxenX-RDBwDwqjiD2Zg/view?usp=share_link). 

Then you can directly run `bash read_dev.sh` or `bash read_test.sh` to get the answers.

### Train retriever

retriever step1 `bash retrieve1.sh`

retriever step2 `bash retrieve2.sh`

### Train reader

`bash read.sh`




<!-- 
## Train Model
... 之后再把完整pipeline写清楚，目前跑如下代码就可以

### Retrieve stage1  (尝试一下就行)
```bash
python retrieve_new_loss.py
```

### Read model （尝试一下就行）
```bash
python read_bart_new.py
``` -->
