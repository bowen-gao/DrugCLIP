# DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/xxxx/blob/main/LICENSE)
[![ArXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2310.06367-B31B1B.svg)](https://arxiv.org/pdf/2310.06367.pdf)

<!-- [[Code](xxxx - Overview)] -->

![cover](framework.png)

Official code for the paper "DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening", accepted at *Neural Information Processing Systems, 2023*. **Currently the code is a raw version, will be updated ASAP**. If you have any inquiries, feel free to contact billgao0111@gmail.com

# Requirements

same as [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol)

**rdkit version should be 2022.9.5**

## Data and checkpoints

https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing

It currently includes the train data, the trained checkpoint and the test data for DUD-E

## Data preprocessing

see py_scripts

## HomoAug

Please refer to HomoAug directory for details

## Train

bash drugclip.sh

## Test

bash test.sh


## Retrieval 

bash retrieval.sh

In the google drive folder, you can find example file for pocket.lmdb and mols.lmdb under retrieval dir.


## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{gao2023drugclip,
    author = {Gao, Bowen and Qiang, Bo and Tan, Haichuan and Jia, Yinjun and Ren, Minsi and Lu, Minsi and Liu, Jingjing and Ma, Wei-Ying and Lan, Yanyan},
    title = {DrugCLIP: Contrasive Protein-Molecule Representation Learning for Virtual Screening},
    booktitle = {NeurIPS 2023},
    year = {2023},
    url = {https://openreview.net/forum?id=lAbCgNcxm7},
}
```
