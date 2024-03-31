import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import random

import lmdb
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem.AllChem as AllChem
import torch
import tqdm
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser()
parser.add_argument('--mol_data_path', type=str, default='/data/protein/DUD-E/raw/all')
parser.add_argument('--lmdb_path', type=str, default='docked_dude_fromweb2D.lmdb')
args = parser.parse_args()


def gen_conformation(mol, num_conf=20, num_worker=8):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, numThreads=num_worker, pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=False)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        except:
            pass
        mol = Chem.RemoveHs(mol)
    except:
        print("cannot gen conf", Chem.MolToSmiles(mol))
        return None
    if mol.GetNumConformers() == 0:
        print("cannot gen conf", Chem.MolToSmiles(mol))
        return None
    return mol

def convert_2Dmol_to_data(mol, num_conf=1, num_worker=5):
    #to 3D
    mol = gen_conformation(mol, num_conf, num_worker)
    if mol is None:
        return None
    coords = [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())]
    atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
    return {'coords': coords, 'atom_types': atom_types, 'smi': Chem.MolToSmiles(mol), 'mol': mol}

def convert_3Dmol_to_data(mol):

    if mol is None:
        return None
    coords = [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())]
    atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
    return {'coords': coords, 'atom_types': atom_types, 'smi': Chem.MolToSmiles(mol), 'mol': mol}

def read_pdb(path):
    pdb_df = PandasPdb().read_pdb(path)

    coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]
    atom_type = pdb_df.df['ATOM']['atom_name']
    residue_name = pdb_df.df['ATOM']['chain_id'] + pdb_df.df['ATOM']['residue_number'].astype(str)
    residue_type = pdb_df.df['ATOM']['residue_name']
    protein = {'coord': np.array(coord), 
               'atom_type': list(atom_type),
               'residue_name': list(residue_name),
               'residue_type': list(residue_type)}
    return protein


def read_sdf_gz_3d(path):
    inf = gzip.open(path)
    with Chem.ForwardSDMolSupplier(inf, removeHs=False, sanitize=False) as gzsuppl:
        ms = [add_charges(x) for x in gzsuppl if x is not None]
    ms = [rdMolStandardize.Uncharger().uncharge(Chem.RemoveHs(m)) for m in ms if m is not None]
    return ms

def add_charges(m):
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
            if at.GetAtomicNum()==6 and at.GetExplicitValence()==5:
                #remove a bond
                for b in at.GetBonds():
                    if b.GetBondType()==Chem.rdchem.BondType.DOUBLE:
                        b.SetBondType(Chem.rdchem.BondType.SINGLE)
                        break
            if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==3:
                at.SetFormalCharge(1)
            if at.GetAtomicNum()==5 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(-1)
    try:
        Chem.SanitizeMol(m)
    except:
        return None
    return m

def get_different_raid(protein, ligand, raid=6):
    protein_coord = protein['coord']
    ligand_coord = ligand['coord']
    protein_residue_name = protein['residue_name']
    pocket_residue = set()
    for i in range(len(protein_coord)):
        for j in range(len(ligand_coord)):
            if np.linalg.norm(protein_coord[i] - ligand_coord[j]) < raid:
                pocket_residue.add(protein_residue_name[i])
    return pocket_residue

def read_mol2_ligand(path):
    mol2_df = PandasMol2().read_mol2(path)
    coord = mol2_df.df[['x', 'y', 'z']]
    atom_type = mol2_df.df['atom_name']
    ligand = {'coord': np.array(coord), 'atom_type': list(atom_type), 'mol': Chem.MolFromMol2File(path)}
    return ligand

def read_smi_mol(path):
    with open(path, 'r') as f:
        mols_lines = list(f.readlines())
    smis = [l.split(' ')[0] for l in mols_lines]
    mols = [Chem.MolFromSmiles(m) for m in smis]
    return mols

def parser(protein_path, mol_path, ligand_path, activity, pocket_index, raid=6):
    protein = read_pdb(protein_path)
    data_mols = read_smi_mol(mol_path)

    ligand = read_mol2_ligand(ligand_path)
    pocket_residue = get_different_raid(protein, ligand, raid=raid)
    pocket_atom_idx = [i for i, r in enumerate(protein['residue_name']) if r in pocket_residue]
    pocket_atom_type = [protein['atom_type'][i] for i in pocket_atom_idx]
    pocket_coord = [protein['coord'][i] for i in pocket_atom_idx]
    pocket_residue_type = [protein['residue_type'][i] for i in pocket_atom_idx]
    pocket_name = protein_path.split('/')[-2]
    pool = mp.Pool(32)
    #mols = [convert_2Dmol_to_data(m) for m in data_mols if m is not None]
    data_mols = [m for m in data_mols if m is not None]
    mols = [m for m in tqdm.tqdm(pool.imap_unordered(convert_2Dmol_to_data, data_mols))]
    mols = [m for m in mols if m is not None]
    
    return [{'atoms': m['atom_types'], 
            'coordinates': m['coords'], 
            'smi': m['smi'],
            'mol': ligand,
            'pocket_name': pocket_name,
            'pocket_index': pocket_index,
            'activity': activity, 
            "pocket_atom_type": pocket_atom_type, 
            "pocket_coord": pocket_coord} for m in mols]

def mol_parser(mol_path, ligand_path, label):
    data_mols = read_smi_mol(mol_path)
    data_mols = [m for m in data_mols if m is not None]  
    ligand = read_mol2_ligand(ligand_path)
    pool = mp.Pool(32) 
    mols = [m for m in tqdm.tqdm(pool.imap_unordered(convert_2Dmol_to_data, data_mols))]
    mols = [m for m in mols if m is not None]
    return [{'atoms': m['atom_types'], 
            'coordinates': m['coords'], 
            'smi': m['smi'],
            'mol': m['mol'],
            'label': label
            } for m in mols]

def pocket_parser(protein_path, ligand_path, pocket_index, raid=6):
    protein = read_pdb(protein_path)
    ligand = read_mol2_ligand(ligand_path)
    pocket_residue = get_different_raid(protein, ligand, raid=raid)
    pocket_atom_idx = [i for i, r in enumerate(protein['residue_name']) if r in pocket_residue]
    pocket_atom_type = [protein['atom_type'][i] for i in pocket_atom_idx]
    pocket_coord = [protein['coord'][i] for i in pocket_atom_idx]
    pocket_residue_type = [protein['residue_type'][i] for i in pocket_atom_idx]
    pocket_name = protein_path.split('/')[-2]
    return {'pocket': pocket_name,
            'pocket_index': pocket_index,
            "pocket_atoms": pocket_atom_type, 
            "pocket_coordinates": pocket_coord}

def write_lmdb(data, lmdb_path):
    #resume

    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    num = 0
    with env.begin(write=True) as txn:
        for d in data:
            txn.put(str(num).encode('ascii'), pickle.dumps(d))
            num += 1

if __name__ == '__main__':
    protein_path = [os.path.join(args.mol_data_path, x, 'receptor.pdb') for x in os.listdir(args.mol_data_path)]
    act_mol_path = [os.path.join(args.mol_data_path, x, 'actives_final.ism') for x in os.listdir(args.mol_data_path)]
    decoy_mol_path = [os.path.join(args.mol_data_path, x, 'decoys_final.ism') for x in os.listdir(args.mol_data_path)]
    
    
    
    for i, pocket in tqdm.tqdm(enumerate(protein_path)):
        # acive mols
        print(i, pocket)
        data = []
        d_active = (mol_parser(act_mol_path[i], pocket.replace('receptor.pdb', 'crystal_ligand.mol2'), 1))
        
        data.extend(d_active)

        # decoy mols
        d_decoy = (mol_parser(decoy_mol_path[i], pocket.replace('receptor.pdb', 'crystal_ligand.mol2'), 0))
        
        data.extend(d_decoy)

        write_lmdb(data, pocket.replace('receptor.pdb', 'mols.lmdb'))

        # write pocket
        d = pocket_parser(pocket, pocket.replace('receptor.pdb', 'crystal_ligand.mol2'), i)
        write_lmdb([d], pocket.replace('receptor.pdb', 'pocket.lmdb'))

        # number of lines in actives_final.smi 
        with open(act_mol_path[i], 'r') as f:
            mols_lines = list(f.readlines())
            print("active", len(d_active), len(mols_lines))


       
        with open(decoy_mol_path[i], 'r') as f:
            mols_lines = list(f.readlines())
            print("decoy", len(d_decoy), len(mols_lines))
        

