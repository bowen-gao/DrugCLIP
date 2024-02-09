import random
import numpy as np
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.Descriptors as Descriptors
import tqdm

def change_torsion(pointset, index, angle=None):
    # pointset: (N, 3)
    #return the new pointset
    # angle: rad
    # index: (2)
    if angle is None:
        angle = random.uniform(-np.pi, np.pi)
    pointset = np.array(pointset)
    #get rotate axis
    axis = pointset[index[0][-1]] - pointset[index[1][0]]
    axis = axis / np.linalg.norm(axis)
    #get rotate matrix
    cos = np.cos(angle)
    sin = np.sin(angle)
    R = np.array([[cos + axis[0] ** 2 * (1 - cos), axis[0] * axis[1] * (1 - cos) - axis[2] * sin, axis[0] * axis[2] * (1 - cos) + axis[1] * sin],
                    [axis[1] * axis[0] * (1 - cos) + axis[2] * sin, cos + axis[1] ** 2 * (1 - cos), axis[1] * axis[2] * (1 - cos) - axis[0] * sin],
                    [axis[2] * axis[0] * (1 - cos) - axis[1] * sin, axis[2] * axis[1] * (1 - cos) + axis[0] * sin, cos + axis[2] ** 2 * (1 - cos)]])
    #rotate
    pointset[index[1]] = np.dot(pointset[index[1]] - pointset[index[1][0]], R) + pointset[index[1][0]]
    return pointset, angle

def gen_conformation(mol, num_conf=20, num_worker=8):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, numThreads=num_worker, pruneRmsThresh=0.1, maxAttempts=5, useRandomCoords=False)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        mol = Chem.RemoveHs(mol)
        return mol
    except:
        return None

def RotatableBond(mol):
    """Get the rotatable bond index of a molecule.
    Args:
        mol: rdkit.Chem.rdchem.Mol
    Returns:
        rotatable_bond: list of tuple
    """
    rotatable_bond = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            rotatable_bond.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return rotatable_bond