## Overview
This repository contains a PyTorch implementation of the models for Hongyu Qian's first rotational project at SABS Oxford with Lhasa Limited.
The pipeline is based on models propsed by other researchers:
- BERT (https://arxiv.org/abs/1810.04805)
- ALBERT (https://arxiv.org/abs/1909.11942)   
- Partition Filter Network (https://aclanthology.org/2021.emnlp-main.17/)
- BioLinkBERT (https://arxiv.org/abs/2203.15827)

## Requirements
The experiments were carried out using the University of Oxford Advanced Research Computing (ARC) facility (http://dx.doi.org/10.5281/zenodo.22558). The dependency packages can be installed with the command:
```
pip install -r requirements.txt
```
Other configurations we use are:  
* python == 3.7.10
* cuda == 11.1
* cudnn == 8
## Data preprocessing
This work uses ***ADE*** and ***n2c2*** datasets and preprocessed to remvoe overlapping entities or annotation errors. Note our classification model is trained and evaluated on ***ADE***. For NER and RE, ***ADE*** corpus is still used in both training and evaluation while ***n2c2*** can be added to the training data as an enrichment.
## Model Training
The training command-line for the pipeline is listed below:  
```bash
python main.py 
--data ${both/ADE}
--embed_mode ${bert_cased/albert/scibert/biolinkbert/sapbert}
--epoch_c 10
--epoch 50
--batch_size_c 20
--batch_size 5
--lr 0.00002
--output_file ${the name of your output files, e.g. ade_test}
```

We can also train the classification or PFN (joint eneity and relation extraction) model on its own using similar command-lines:
```bash
python classification.py 
--embed_mode ${bert_cased/albert/scibert/biolinkbert/sapbert}
--epoch 10
--batch_size 20
--lr 0.00002
--dropout 0.1
--output_file ${the name of your output files, e.g. ade_classification}
```
and
```bash
python PFN.py
--data ${both/ADE}
--embed_mode ${bert_cased/albert/scibert/biolinkbert/sapbert}
--epoch 50
--batch_size 5
--lr 0.00002
--dropout 0.1
--dropconnect 0.1
--hidden_size 300
--clip 0.25
--output_file ${the name of your output files, e.g. ade_NER_RE}
```
### Inference on Customized Input
For infernece on new inputs with the trained model, please run the following code:
```bash
python inference.py
--model_file ${name of the trained model}
--text ${path to the text file you want to infer}
--embed_mode ${bert_cased/albert/scibert/biolinkbert/sapbert}
```
The output is a json file containing all sentences with predicted entities and relations