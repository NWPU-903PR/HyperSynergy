import pandas as pd
import numpy as np
import os
from rdkit import Chem
import networkx as nx
import json


def SDF2CSV(Input_path, Outputpath):
    suppl = Chem.SDMolSupplier(Input_path)
    molss = [x for x in suppl if x is not None]
    smiles = [Chem.MolToSmiles(mol) for mol in molss]
    y_name = '_Name'
    y = pd.DataFrame([mol.GetProp(y_name) for mol in molss])
    print(y)
    y.index = smiles
    y.columns = [y_name]
    dataset = y
    dataset.to_csv(Outputpath)


def CSV2JSON(fpath, Input_file, Output_file, flag):
    infile = open(os.path.join(fpath, Input_file), 'r')
    data1 = []
    i = 0
    for line in infile:
        data = line.rstrip().split(",")  # ("\t")
        if i > 0:
            data1.append(data)
        i = i + 1
    date_information = np.array(data1)
    list1, list2 = [], []
    if flag == "drug":
        list1 = (date_information[:, 1]).tolist()
        list2 = (date_information[:, 0]).tolist()
    elif flag == "cell line":
        list1 = (date_information[:, 1]).tolist()  # 2
        list2 = (date_information[:, 2:]).tolist()  # 3

    data_direction = dict(zip(list1, list2))
    jsonsmile = json.dumps(data_direction)
    filename = open(os.path.join(fpath, Output_file), 'w')
    filename.write(jsonsmile)

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size=mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def geneate_direction(fpath,Input_file,Output_file):
    #fpath = 'data/'
    drugs = json.load(open(os.path.join(fpath,Input_file)))
    smile_graph = {}
    for d in drugs.keys():
        G = {'c_size': [], 'features': [], 'edge_index': []}
        lg = drugs[d]
        c_size, features, edge_index = smile_to_graph(lg)
        print(type(c_size), type(features), type(edge_index))
        G['c_size'].append(c_size)
        G['features'].append(features)
        G['edge_index'].append(edge_index)
        smile_graph[d] = G
        jsongraph = json.dumps(smile_graph)
        filename = open(os.path.join(fpath,Output_file),'w')
        filename.write(jsongraph)
        
if __name__=='__main__':
    # generate drug graphs
    SDF2CSV('data/drug_2d.sdf', 'data/drug_smiles.csv')
    CSV2JSON('data/', 'drug_smiles.csv', 'drug_smiles.json', flag="drug")
    geneate_direction('data/', "drug_smiles.json", "drug_graph.json")

    # gene_expression
    CSV2JSON('data/', 'gene_expression_cell.csv', 'gene_expression_cell.json', flag="cell line")


