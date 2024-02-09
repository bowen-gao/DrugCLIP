# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import lru_cache

import numpy as np
from unicore.data import BaseWrapperDataset

from . import data_utils


class VAEBindingDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        selfies,
        is_train=True,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.selfies = selfies
        self.is_train = is_train
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        coordinates = self.dataset[index][self.coordinates]
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index]["pocket"]
        #affinity = self.dataset[index][self.affinity]
        selfies = np.array(self.dataset[index][self.selfies])
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "selfies": selfies
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class VAEBindingTestDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        is_train=True,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.is_train = is_train
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        coordinates = self.dataset[index][self.coordinates]
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        smi = self.dataset[index]["smi"]
        pocket = self.dataset[index]["pocket_name"]
        lig = self.dataset[index]["lig_name"]
        #affinity = self.dataset[index][self.affinity]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "holo_coordinates": coordinates.astype(np.float32),#placeholder
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "smi": smi,
            "pocket": pocket,
            "lig": lig
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class VAEGenerationTestDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        pocket_atoms,
        pocket_coordinates,
        is_train=True,
    ):
        self.dataset = dataset
        self.seed = seed
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.is_train = is_train
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])

        
        return {
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


