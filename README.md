
# HyperSynergy

Source code and dataset for TPAMI 2023 [paper](https://ieeexplore.ieee.org/abstract/document/10050784/): "Few-Shot Drug Synergy Prediction with a Prior-Guided Hypernetwork Architecture"

## Requirements

Python == 3.6

torch == 1.7.0+cu110

torch-geometric == 1.6.1

torch-scatter == 2.0.5

torch-sparse == 0.6.8

torch-cluster == 1.5.8                                     

rdkit == 2019.09.3.0

pytorchtools == 0.0.2


## Usage (Step by step runing)

### 0. Create Sample Feature Dictionaries （Required when running on other datasets）

If you would like to run the code of the HyperSynergy on other datasets, please first obtain the drugs' sdf file from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) through the drugs' Pubchem cid, and gene expression profile (900 genes, **gene_expression_examples.csv**) of cell lines from [CCLE](https://depmap.org/portal/download/all/) or [GDSC](https://www.cancerrxgene.org/downloads/bulk_download), and then create three .p files: cell_feature_900.p, codes_cell.p, and drug_feature_cell.p by running feature_generation.py and feature_direction.py

### 1. Representation Learning

If you would like to get a pretrained feature embedding model, please run representation_learning.py

### 2. Meta-training 

If you would like to learn the weights and biases of meta-generative network module of the HyperSynergy, please run train_few_shot.py

### 3. Meta-test

If you would like to test the power of HyperSynergy on meta-test cell lines, please run test_few_shot.py

## Citation

If you find our work is useful for your research, please consider citing our paper:

@article{zhang2023few,<br>
  title={Few-Shot Drug Synergy Prediction With a Prior-Guided Hypernetwork Architecture},<br>
  author={Zhang, Qingqing and Zhang, Shaowu and Feng, Yuehua and Shi, Jianyu},<br>
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},<br>
  year={2023},<br>
  publisher={IEEE}
}
