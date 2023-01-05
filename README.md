# Relation-aware Graph Attention Network for Visual Question Answering

An unofficial Tensorflow 2.x based implementation of [Relation-Aware Graph Attention Network for Visual Question Answering](https://arxiv.org/pdf/1903.12314.pdf), ICCV 2019 paper.

This is re-write of PyToch 1.0.1 based implementation available [here](https://github.com/linjieli222/VQA_ReGAT). Some parts are work in progress (explicit relation encoder, semantic relation encoder, BAN and MuTAN). You can train BUTD based  model.

## Environment

- Tensorflow 2.x


## Getting started

### 1. Download data

```
source download.sh
```   

The total size of data is about 90GB, and the structure of dataset is as follows.

```
├── data
│   ├── Answers
│   ├── Bottom-up-features-adaptive
│   ├── Bottom-up-features-fixed
│   ├── cache
│   ├── cp_v2_annotations
│   ├── cp_v2_questions
│   ├── glove
│   ├── imgids
│   ├── Questions
│   ├── visualGenome
```

### 2. Train the model

	python main.py --config config/butd_vqa.json
    
I trained the model with A100 40GB GPU (batch size: 256), and the code takes about 39GB GPU RAM.
    
### 3. Evaluate the model

	python main.py --config config/butd_vqa.json  --mode eval --checkpoint <pretrained_model_path>
    
## Result

| | Accuracy (BUTD fusion) |
|-|-|
|Official PyTorch Code| 63.99|
|Tensorflow 2.0 Code| 63.24 |


