
# HyperSynergy

## Requirements

Python 3.6

torch 1.7.0+cu110

torch-geometric 

torch-scatter 

torch-sparse  

torch-cluster 

rdkit 

pytorchtools 



## Usage (Step by step runing)

### 0. Create Dictionaries

please create files: cell_feature_900.p, codes_cell.p and drug_feature_cell.p

### 1. Representation Learning

please run representation_learning.py

### 2. Meta-training 

please run train_few_shot.py

### 3. Meta-test

please run test_few_shot.py
