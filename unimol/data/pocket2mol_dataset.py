from functools import lru_cache
import sys
import pickle
import random
import networkx as nx
import numpy as np
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append('..')

import numpy as np
from unicore.data import BaseWrapperDataset

from . import data_utils
from unimol.utils import geom

def gen_conformation(mol, num_conf=20, num_worker=8, keepHs=False):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, numThreads=num_worker, pruneRmsThresh=0.1, maxAttempts=5, useRandomCoords=False)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        if not keepHs:
            mol = Chem.RemoveHs(mol)
        return mol
    except:
        return None

class FragmentConformationDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        vocab,
        conf_vocab,
        use_pocket=True,
        is_train=True
    ):
        self.dataset = dataset
        self.seed = seed
        self.use_pocket = use_pocket
        self.conf_vocab = Vocabulary(vocab, conf_vocab)
        self.is_train = is_train
        self.set_epoch(None)
    
    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
        

    def parse_frag_mol(self, frag_mol):
        atom_types = [a.GetSymbol() for a in frag_mol.GetAtoms()]
        atom_coords = np.array(frag_mol.GetConformer(0).GetPositions())
        return {'atom_types': atom_types, 'atom_coords': atom_coords}
    
    def parse_frag_idx(self, vocab_conf, full_mol, atom_map):
        if vocab_conf.GetNumConformers() == 0:
            smi = Chem.MolToSmiles(vocab_conf)
            vocab_conf = Chem.MolFromSmiles(smi)
            vocab_conf = gen_conformation(vocab_conf, num_conf=1, num_worker=1, keepHs=True)
        mol = Chem.RWMol(full_mol)
        atom_idx = list(range(full_mol.GetNumAtoms()))
        for i, atom in enumerate(full_mol.GetAtoms()):
            if atom.GetAtomMapNum() not in atom_map:
                atom_idx[i] = -1
        for i in range(len(atom_idx) - 1, -1, -1):
            if atom_idx[i] == -1:
                mol.RemoveAtom(i)
        mol = mol.GetMol()
        #mol = Chem.RemoveHs(mol)
        smi = Chem.MolToSmiles(mol)
        #find the map num in smiles
        map_num = []
        smi_p = smi.split('[')
        for i in range(1, len(smi_p)):
            if ':' in smi_p[i]:
                end_idx = smi_p[i].split(':')[1].index(']')
                map_num.append(int(smi_p[i].split(':')[1][:end_idx]))

        vocab_conf = Chem.RemoveHs(vocab_conf)
        for i, atom in enumerate(vocab_conf.GetAtoms()):
            if atom.GetSymbol() != 'H':
                atom.SetAtomMapNum(map_num[i])
        vocab_conf = Chem.AddHs(vocab_conf, addCoords=True)

        if torch.isnan(torch.from_numpy(np.array(vocab_conf.GetConformer(0).GetPositions()))).any():
            vocab_conf = gen_conformation(vocab_conf, num_conf=1, num_worker=1, keepHs=True)

        return vocab_conf


    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]
    
    def check_leaf(self, edges, index):
        out_degree = 0
        for edge in edges:
            if edge[0] == index:
                out_degree += 1
        if out_degree == 0:
            return True
        else:
            return False

    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]
    
    def check_leaf(self, edges, index):
        out_degree = 0
        for edge in edges:
            if edge[0] == index:
                out_degree += 1
        if out_degree == 0:
            return True
        else:
            return False

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        
        random.seed(self.seed + epoch)
        #pocket
        if self.use_pocket:
            pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index]['pocket_atom']]
        )
        pocket_coordinates = np.stack(self.dataset[index]['pocket_coord'])

        full_mol = self.dataset[index]['frags']['mol']

        frag_graph = nx.Graph()
        edges = self.dataset[index]['frags']['frag_edges']
        frag_graph.add_edges_from(np.array(edges))
        if len(edges) == 0:
            frag_mol_idx = self.dataset[index]['frags']['frag_idx'][0]
            frag_mol = self.conf_vocab.conf[frag_mol_idx]
            f_mol_noH_num = len([atom for atom in frag_mol.GetAtoms() if atom.GetSymbol() != 'H'])
            frag_mol = Chem.AddHs(frag_mol, addCoords=True)
            frag_mol_data = self.parse_frag_mol(frag_mol)
            return {
                'atom_types': np.array([], dtype=str),
                'atom_coords': np.zeros((1, 3), dtype=np.float32),
                'focal_atom_local': 0,#place holder
                'attach_atom_local': 0,#place holder
                'focal_atom': 0,#place holder
                'attach_atom': 0,#place holder
                'frag_atom_types': frag_mol_data['atom_types'],
                'frag_atom_coords': frag_mol_data['atom_coords'],
                'end': True,
                'torsion_output_prev': 0,#place holder
                'coords_input_prev': frag_mol_data['atom_coords'],
                'atom_types_withfocal': frag_mol_data['atom_types'],
                'pocket_atoms': pocket_atoms,
                'pocket_coordinates': pocket_coordinates,
                'first': True,
                'symmetric': True
            }
        bfs_edges = list(nx.bfs_edges(frag_graph, 0))
        #reorder the link and local link
        link = self.dataset[index]['frags']['links']
        local_link = self.dataset[index]['frags']['links_local']
        reorrder_link, reorrder_local_link = [], []
        for i, b_edge in enumerate(bfs_edges):
            for j, o_edge in enumerate(edges):
                if b_edge[0] == o_edge[0] and b_edge[1] == o_edge[1]:
                    reorrder_link.append(link[j])
                    reorrder_local_link.append(local_link[j])
                elif b_edge[0] == o_edge[1] and b_edge[1] == o_edge[0]:
                    reorrder_link.append([link[j][1], link[j][0]])
                    reorrder_local_link.append([local_link[j][1], local_link[j][0]])
        self.dataset[index]['frags']['links'] = reorrder_link
        self.dataset[index]['frags']['links_local'] = reorrder_local_link
        #clip a random subgraph
        bfs_edges_full = bfs_edges.copy()
        clip_step = random.randint(0, len(bfs_edges))
        #print('clip_step', clip_step)
        start_frag = bfs_edges[0][0]
        if clip_step != 0:
            bfs_edges = bfs_edges[:clip_step]
            focal_frag, attach_frag = bfs_edges[-1]
        else:
            focal_frag, attach_frag = bfs_edges[0]

        for i, atom in enumerate(full_mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
        
        end = (clip_step == len(bfs_edges_full))
        if clip_step == 0:
            #frag
            #print('first frag')
            frag_mol_idx = self.dataset[index]['frags']['frag_idx'][start_frag]
            frag_mol = self.conf_vocab.conf[frag_mol_idx]
            if frag_mol is None:
                print(frag_mol_idx, 'is None')
            frag_mol = self.parse_frag_idx(frag_mol, full_mol, self.dataset[index]['frags']['map'][start_frag])
            frag_mol_data = self.parse_frag_mol(frag_mol)
            return {
                'atom_types': np.array([], dtype=str),
                'atom_coords': np.zeros((1, 3), dtype=np.float32),
                'focal_atom_local': 0,#place holder
                'attach_atom_local': 0,#place holder
                'focal_atom': 0,#place holder
                'attach_atom': 0,#place holder
                'frag_atom_types': frag_mol_data['atom_types'],
                'frag_atom_coords': frag_mol_data['atom_coords'],
                'end': end,
                'torsion_output_prev': 0,#place holder
                'coords_input_prev': frag_mol_data['atom_coords'],
                'atom_types_withfocal': frag_mol_data['atom_types'],
                'pocket_atoms': pocket_atoms,
                'pocket_coordinates': pocket_coordinates,
                'first': True,
                'symmetric': True
            }
        
        
        clip_frag_idx = [e[0] for e in bfs_edges] + [e[1] for e in bfs_edges[:-1]]
        clip_frag_idx = np.unique(clip_frag_idx)
        frag_attach_idx = bfs_edges[-1][1]
        clip_map = []
        for i in range(len(clip_frag_idx)):
            clip_map.extend(self.dataset[index]['frags']['map'][clip_frag_idx[i]])
        clip_map_attach = clip_map + self.dataset[index]['frags']['map'][frag_attach_idx]
        #print('frag_node_map', self.dataset[index]['frags']['map'])

        #get part mol
        part_mol = Chem.RWMol(full_mol)
        atom_idx = list(range(full_mol.GetNumAtoms()))
        for i, atom in enumerate(full_mol.GetAtoms()):
            if atom.GetSymbol() == 'H':
                #remove if neighbor is not in clip_map
                neighbor_atom = [atom.GetAtomMapNum() for atom in part_mol.GetAtomWithIdx(i).GetNeighbors()]
                neighbor_map = [atom.GetAtomMapNum() for atom in part_mol.GetAtomWithIdx(i).GetNeighbors()][0]
                if neighbor_map not in clip_map:
                    atom_idx[i] = -1
                if atom.GetAtomMapNum() not in clip_map:
                    atom_idx[i] = -1
            else:
                if atom.GetAtomMapNum() not in clip_map:
                    atom_idx[i] = -1
        for i in range(len(atom_idx) - 1, -1, -1):
            if atom_idx[i] == -1:
                part_mol.RemoveAtom(i)
        
        frag_exp_link = []
        for i in range(len(clip_frag_idx)):
            for e_d, e in enumerate(bfs_edges_full):
                if e[0] == clip_frag_idx[i] and e[1] not in clip_frag_idx:
                    frag_exp_link.append(self.dataset[index]['frags']['links'][e_d][0])
        for link_mp in frag_exp_link:
            add_map = [i for i, atom in enumerate(part_mol.GetAtoms()) if atom.GetAtomMapNum() == link_mp][0]
            part_mol.AddAtom(Chem.Atom(1))
            part_mol.AddBond(add_map, part_mol.GetNumAtoms() - 1, Chem.rdchem.BondType.SINGLE)
        part_mol = part_mol.GetMol()
        part_mol = Chem.RemoveHs(part_mol)
        part_mol = Chem.AddHs(part_mol, addCoords=True)


        part_mol_atom_types = [atom.GetSymbol() for atom in part_mol.GetAtoms()]
        part_mol_atom_coords = np.array([part_mol.GetConformer().GetAtomPosition(i) for i in range(part_mol.GetNumAtoms())])

        #get part mol with attach
        part_mol_attach = Chem.RWMol(full_mol)
        atom_idx = list(range(full_mol.GetNumAtoms()))
        for i, atom in enumerate(full_mol.GetAtoms()):
            if full_mol.GetAtomWithIdx(i).GetSymbol() == 'H':
                #remove if neighbor is not in clip_map
                neighbor = [atom.GetAtomMapNum() for atom in part_mol_attach.GetAtomWithIdx(i).GetNeighbors()][0]
                if neighbor not in clip_map_attach:
                    atom_idx[i] = -1
            else:
                if atom.GetAtomMapNum() not in clip_map_attach:
                    atom_idx[i] = -1
        #print([i for i, idx in enumerate(atom_idx) if idx == -1])
        for i in range(len(atom_idx) - 1, -1, -1):
            if atom_idx[i] == -1:
                part_mol_attach.RemoveAtom(i)
        #if not self.check_leaf(edges, frag_attach_idx):
        frag_exp_link = []
        clip_frag_attach_idx = list(clip_frag_idx) + [frag_attach_idx]
        for i in range(len(clip_frag_attach_idx)):
            for e_d, e in enumerate(bfs_edges_full):
                if e[0] == clip_frag_attach_idx[i] and e[1] not in clip_frag_attach_idx:
                    frag_exp_link.append(self.dataset[index]['frags']['links'][e_d][0])
        for link_mp in frag_exp_link:
            add_map = [i for i, atom in enumerate(part_mol_attach.GetAtoms()) if atom.GetAtomMapNum() == link_mp][0]
            part_mol_attach.AddAtom(Chem.Atom(1))
            part_mol_attach.AddBond(add_map, part_mol_attach.GetNumAtoms() - 1, Chem.rdchem.BondType.SINGLE)
            #print('add H atom symbol', part_mol_attach.GetAtomWithIdx(add_map).GetSymbol())
        #else:
        #    print('leaf node')
        part_mol_attach = part_mol_attach.GetMol()
        part_mol_attach = Chem.RemoveHs(part_mol_attach)
        part_mol_attach = Chem.AddHs(part_mol_attach, addCoords=True)

        part_mol_attach_atom_types = [atom.GetSymbol() for atom in part_mol_attach.GetAtoms()]
        part_mol_attach_atom_coords = np.array([part_mol_attach.GetConformer().GetAtomPosition(i) for i in range(part_mol_attach.GetNumAtoms())])
        '''
        part_mol_atom = [self.dataset[index]['frags']['map'][e[0]] for e in bfs_edges] + \
                        [self.dataset[index]['frags']['map'][e[1]] for e in bfs_edges[:-1]]
        part_mol_atom = np.concatenate(part_mol_atom, axis=0)
        part_mol_atom = np.unique(part_mol_atom)
        part_mol_atom_types = self.dataset[index]['atom_types'][part_mol_atom]
        part_mol_atom_coords = self.dataset[index]['atom_coords'][part_mol_atom]
        '''

        '''
        #add focal atom
        part_mol_atom_withfocal = [self.dataset[index]['frags']['map'][e[0]] for e in bfs_edges] + \
                        [self.dataset[index]['frags']['map'][e[1]] for e in bfs_edges]
        part_mol_atom_withfocal = np.concatenate(part_mol_atom_withfocal, axis=0)
        part_mol_atom_withfocal = np.unique(part_mol_atom_withfocal)
        part_mol_atom_types_withfocal = self.dataset[index]['atom_types'][part_mol_atom_withfocal]
        part_mol_atom_coords_withfocal = self.dataset[index]['atom_coords'][part_mol_atom_withfocal]
        '''
        focal_atom_local = [i for i, atom in enumerate(part_mol.GetAtoms()) if atom.GetAtomMapNum() == self.dataset[index]['frags']['links'][clip_step - 1][0]][0]
        focal_atom = [i for i, atom in enumerate(part_mol_attach.GetAtoms()) if atom.GetAtomMapNum() == self.dataset[index]['frags']['links'][clip_step - 1][0]][0]
        #focal_atom = self.dataset[index]['frags']['map'][focal_frag][focal_atom_local]
        
        #frag
        frag_mol_idx = self.dataset[index]['frags']['frag_idx'][attach_frag]
        frag_mol = self.conf_vocab.conf[frag_mol_idx]

        frag_mol = self.parse_frag_idx(frag_mol, self.dataset[index]['frags']['mol'], self.dataset[index]['frags']['map'][attach_frag])
        frag_mol_data = self.parse_frag_mol(frag_mol)

        if frag_mol is None:
                print(frag_mol_idx, 'is None')

        attach_atom_local = [i for i, atom in enumerate(frag_mol.GetAtoms()) if atom.GetAtomMapNum() == self.dataset[index]['frags']['links'][clip_step - 1][1]][0]
        attach_atom = [i for i, atom in enumerate(part_mol_attach.GetAtoms()) if atom.GetAtomMapNum() == self.dataset[index]['frags']['links'][clip_step - 1][1]][0]
        #attach_atom = self.dataset[index]['frags']['map'][bfs_edges[-1][1]][attach_atom_local]


        #torsion angles
        #prev_edge = bfs_edges[-1]
        '''
        prev_link_local = self.dataset[index]['frags']['link_atoms'][bfs_edges.index(prev_edge) - 1]
        prev_link = (self.dataset[index]['frags']['map'][prev_edge[0]][prev_link_local[0]], 
                self.dataset[index]['frags']['map'][prev_edge[1]][prev_link_local[1]])
        index_rotate = [prev_link[1]] + self.dataset[index]['frags']['map'][focal_frag]
        index_parent = np.concatenate([self.dataset[index]['frags']['map'][e[0]] for e in bfs_edges[:-1]], axis=0) + [prev_link[0]]
        '''
        index_rotate = [i for i, atom in enumerate(part_mol_attach.GetAtoms()) if atom.GetAtomMapNum() in self.dataset[index]['frags']['map'][attach_frag]]
        index_rotate.remove(attach_atom)
        index_rotate = [attach_atom] + index_rotate
        index_parent = []
        for e in bfs_edges[:-1]:
            index_parent += self.dataset[index]['frags']['map'][e[0]]
            index_parent += self.dataset[index]['frags']['map'][e[1]]
        index_parent += self.dataset[index]['frags']['map'][focal_frag]
        index_parent = list(np.unique(index_parent))
        index_parent = [i for i, atom in enumerate(part_mol_attach.GetAtoms()) if atom.GetAtomMapNum() in index_parent]
        index_parent.remove(focal_atom)
        index_parent = index_parent + [focal_atom]
        #add hydrogen
        # get the hydrogen that connects to the rotate atoms
        index_rotate_h = []
        for i in index_rotate:
            for j in part_mol_attach.GetAtomWithIdx(i).GetNeighbors():
                if j.GetAtomicNum() == 1:
                    index_rotate_h.append(j.GetIdx())
        index_rotate += index_rotate_h
        # get the hydrogen that connects to the parent atoms
        index_parent_h = []
        for i in index_parent:
            for j in part_mol_attach.GetAtomWithIdx(i).GetNeighbors():
                if j.GetAtomicNum() == 1:
                    index_parent_h.append(j.GetIdx())
        index_parent = index_parent_h + index_parent
        coords_input_prev, torsion_output_prev = geom.change_torsion(part_mol_attach_atom_coords, [index_parent, index_rotate])
        symmetric = (len(index_rotate) == 0 )

        return {
            'atom_types': part_mol_atom_types,
            'atom_coords': part_mol_atom_coords,
            'focal_atom_local': focal_atom_local,
            'attach_atom_local': attach_atom_local,
            'focal_atom': focal_atom,
            'attach_atom': attach_atom,
            'frag_atom_types': frag_mol_data['atom_types'],
            'frag_atom_coords': frag_mol_data['atom_coords'],
            'end': end,
            'torsion_output_prev': torsion_output_prev,
            'coords_input_prev': coords_input_prev,
            'atom_types_withfocal': part_mol_attach_atom_types,
            'pocket_atoms': pocket_atoms,
            'pocket_coordinates': pocket_coordinates,
            'first': False,
            'symmetric': symmetric
        }
    
    def __getitem__(self, index: int):
        item = self.__cached_item__(index, self.epoch)
        return item
    