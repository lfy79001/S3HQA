# S3HQA

## requirements
```bash
python==3.7
torch==1.7.1+cu110
transformers==4.21.1
```

## Data prepare
Download all data from [hear](【超级会员V2】我通过百度网盘分享的文件：Data.zip
链接：https://pan.baidu.com/s/17rK9CaIz461BluEBwM91xg 
提取码：LJkB 
复制这段内容打开「百度网盘APP即可获取」). Then `unzip Data.zip`
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
