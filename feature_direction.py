import numpy as np
from utilities import Mapping
import pandas as pd
import torch
import json
import os
import pickle

def gene_feature(GeneDataset0):
    GeneDataset = []
    data_len = len(GeneDataset0)
    for i in range(data_len):
        GeneDataset.append(float(GeneDataset0[i]))
    GeneDataset = torch.tensor(GeneDataset,dtype=torch.float)
    return GeneDataset

fpath= 'data/'
# drug pubchem cid
drug_feature=pd.read_table(os.path.join(fpath,"drug_id.txt"),sep = "	",header=None)
#cell lines id
cell_feature1=pd.read_table(os.path.join(fpath,"cell_line_id.txt"),sep = "	")

cell_feature=cell_feature1.loc[:,['CELL_ID']]
drug_feature.columns = ["DRUG_ID"]

drug_feature['DRUG_ID']=drug_feature['DRUG_ID'].apply(lambda x: int(x))
cell_feature['CELL_ID']=cell_feature['CELL_ID'].apply(lambda x: str(x))

codes={'drugs': Mapping(drug_feature['DRUG_ID']),
       'cells':Mapping(cell_feature['CELL_ID'])}

# drugs
DG= json.load(open(fpath + "drug_graph.json"), encoding='utf-8')
drug_feature['c_size']=drug_feature['DRUG_ID'].apply(lambda x: DG[str(int(x))].get('c_size'))
drug_feature['c_size']=drug_feature['c_size'].apply(lambda x: torch.tensor(x,dtype=torch.long))

drug_feature['features']=drug_feature['DRUG_ID'].apply(lambda x: DG[str(int(x))].get('features'))
drug_feature['features']=drug_feature['features'].apply(lambda x: torch.tensor(x,dtype=torch.float))
drug_feature['features']=drug_feature['features'].apply(lambda x: torch.squeeze(x))

drug_feature['edge_index']=drug_feature['DRUG_ID'].apply(lambda x: DG[str(int(x))].get('edge_index'))
drug_feature['edge_index']=drug_feature['edge_index'].apply(lambda x: torch.tensor(x,dtype=torch.long))
drug_feature['edge_index']=drug_feature['edge_index'].apply(lambda x: torch.squeeze(x))


# cell lines
CG = json.load(open(fpath + "gene_expression_cell.json", encoding='utf-8'))
cell_feature['cg']=cell_feature['CELL_ID'].apply(lambda x: CG[str(x)])
cell_feature['cg']=cell_feature['cg'].apply(lambda x: gene_feature(x))

pickle.dump(cell_feature, open(fpath+'cell_feature_900.p', 'wb'))
pickle.dump(drug_feature, open(fpath+'drug_feature_cell.p', 'wb'))
pickle.dump(codes, open(fpath+'codes_cell.p', 'wb'))

