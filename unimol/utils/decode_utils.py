from unimol.data.dictionary import DecoderDictionary
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import MolFromSmiles


def one_hot_to_selfies(hot, dict1:DecoderDictionary):
    '''> 3 means to get rid of special tokens in the molecule representation.'''
    selfies_list = []
    # print(hot.transpose(0, 1).argmax(1))
    for idx in hot.transpose(0, 1).argmax(1):
        if idx.item() == dict1.index('[SEP]') or idx.item() == dict1.index('[PAD]'):
            break
        elif idx.item() == dict1.index('[UNK]') or idx.item() == dict1.index('[CLS]'):
            selfies_list.append('[nop]')            
        else:
            selfies_list.append(dict1.index2symbol(idx.item()))
    # print("selfies_list: {}".format(selfies_list))
    # return ''.join([dict.index2symbol(idx.item()) if idx.item() > 3 else '' for idx in hot.transpose(0, 1).argmax(1)]).replace(' ', '')
    # return ''.join([dict.index2symbol(idx.item()) if idx.item() > 3 else '[nop]' for idx in hot.transpose(0, 1).argmax(1)]).replace(' ', '')
    return ''.join(selfies_list).replace(' ', '')


def one_hot_to_smiles(hot, dict_):
    '''Return both the smile repre. and the selfies rep.'''
    selfies = one_hot_to_selfies(hot, dict_)
    # selfies_list = list(sf.split_selfies(selfies))
    # return sf.decoder(selfies), selfies_list
    return sf.decoder(selfies)


def label_smiles(smiles:list):
    """Label a batch of smiles to in the form of Unimol compatible dataset"""

    selfies = [list(sf.split_selfies(sf.encoder(smile))) for smile in smiles]
    new_data_list = []
    
    for idx, smile in enumerate(smiles):
        data_dict = dict()
        try:
            m = Chem.MolFromSmiles(smile)
            m3d = Chem.AddHs(m)
        except:
            # invalid smile generated
            continue

        atom_list = []
        for atom in m3d.GetAtoms():
            atom_list.append(atom.GetSymbol())
        
        selfie = selfies[idx]

        #print(selfie)

        #selfie_idx = [dict1.index(item) for item in selfie]
        #print(selfie_idx)




        data_dict['atoms'] = atom_list
        
        # coord_list = []
        # cids = AllChem.EmbedMultipleConfs(m3d, numConfs=10, numThreads=0)
        # for id in cids:
        #     conf = m3d.GetConformer(id=id)
        #     coord_list.append(conf.GetPositions())
        # data_dict['coordinates'] = coord_list
        data_dict['coordinates'] = [] # No need to add coordinates
        
        data_dict['smi'] = smile
        data_dict['scaffold'] = ''
        data_dict['ori_index'] = -1
        data_dict['selfies'] = selfies[idx]
        data_dict['target'] = MolLogP(MolFromSmiles(smile))

        new_data_list.append(data_dict)
    
    return new_data_list