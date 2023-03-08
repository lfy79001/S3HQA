# S3HQA

## requirements
python==3.7
torch==1.7.1+cu110
transformers==4.21.1

## Data prepare
Put all data in ./Data/HybridQA
Put bert-large model in ./PTM/bert-large-uncased

## Retrieve stage1
```bash
learning_rate = 7e-6
is_train = 1
is_test = 0
is_generate = 0
is_firststage = 1
```
