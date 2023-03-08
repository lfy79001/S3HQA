# S3HQA

## requirements
```bash
python==3.7
torch==1.7.1+cu110
transformers==4.21.1
```

## Data prepare
Download all data from [hear](). Then `unzip Data.zip`
Put `bert-large model` in `./PTM/bert-large-uncased` and `bart-large model` in  `./PTM/bart-large`

## Train Model
... 之后再把完整pipeline写清楚，目前跑如下代码就可以

### Retrieve stage1  (尝试一下就行)
```bash
python retrieve_new_loss.py
```

### Read model （尝试一下就行）
```bash
python read_bart_new.py
```
