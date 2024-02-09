# Code for HomoAug

Homoaug is a data augmentation method for generate pseudo-pocket-ligand pairs. It is based on the idea that the ligand binding sites of homologous proteins are similar. The details of HomoAug can be found in our paper: "DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening"

## Requirements

To run the code, you need to install the following packages:

- ray (version 1.12.0)
- jackmmer (a part of HMMER)
- esl-reformat (a part of HMMER)
- TM-align 

Please use 'pip install ray==1.12.0' to install ray.

The exeauctable files of jackmmer, esl-reformat and TM-align are included in our folder 'bin', please add the path of 'bin' to your environment variable.

## Usage

<!---     parser = argparse.ArgumentParser()
    parser.add_argument("--id_file", type=str, default="/drug/BioLip/tmp.id")
    parser.add_argument("--homoaug_dir", type=str, default="/drug/BioLip/homoaug_new")
    parser.add_argument(
        "--fasta_file",
        type=str,
        default="/drug/BioLip/BioLiP_v2023-04-13_regularLigand.fasta",
    )
    parser.add_argument(
        "--protein_pdb_dir", type=str, default="/drug/BioLip/protein_pdb"
    )
    parser.add_argument(
        "--pocket_pdbs_dir", type=str, default="/drug/BioLip/pocket_pdb"
    )
    parser.add_argument(
        "--jackhmmer_output_dir", type=str, default="/drug/BioLip/pdbbind_MSA_fasta"
    )
    parser.add_argument(
        "--n_thread", type=int, default=10, help="number of threads for running"
    )
    parser.add_argument(
        "--database_fasta_path",
        type=str,
        default="/data/protein/AF2DB/AFDB_HC_50.fa",
        help="jackhmmer search database, in fasta format",
    )
    parser.add_argument(
        "--database_pdb_dir",
        type=str,
        default="/drug/AFDB_HC_50_PDB",
        help="homoaug search database, e.g. AF2DB",
    )
    parser.add_argument(
        "--max_extend_num",
        type=int,
        default=20,
        help="max number of extended pocket-ligand pairs for one real pocket-ligand pair",
    )
    parser.add_argument(
        "--TMscore_threshold",
        type=float,
        default=0.4,
        help="TMscore threshold for extending",
    )
    parser.add_argument(
        "--Match_rate_threshold",
        type=float,
        default=0.4,
        help="Match_rate threshold for extending",
    )
-->


To use HomoAug, you only need to run the following command:

```bash
    python run_HomoAug.py
        --id_file your_id_file
        --homoaug_dir your_homoaug_dir
        --fasta_file your_fasta_file
        --protein_pdb_dir your_protein_pdb_dir
        --pocket_pdbs_dir your_pocket_pdbs_dir
        --jackhmmer_output_dir your_jackhmmer_output_dir
        --n_thread your_n_thread
        --database_fasta_path your_database_fasta_path
        --database_pdb_dir your_database_pdb_dir
        --max_extend_num your_max_extend_num
        --TMscore_threshold your_TMscore_threshold
        --Match_rate_threshold your_Match_rate_threshold
```

We will explain the meaning of each parameter in the following.


#### --id_file

The id_file is a file containing the ids of the real pocket-ligand pairs. 

For example, the id_file of BioLip dataset can be like this:

```
2WNS_B_receptor_B_550_OMP
5RVK_A_receptor_A_201_2AK
5F03_A_receptor_A_301_5TA
7EXF_B_receptor_B_801_GAL
4OJ4_A_receptor_A_501_DIF
1PJ7_A_receptor_A_2887_FFO
......
```

You need to create your own id_file according to your dataset, with the format like :`<pdb_id>_<chain_id>_<any_string>_<ligand_name_in_pdb>

#### --homoaug_dir

The output directory of HomoAug. When the program is finished, you will find the extended pocket-ligand pairs in this directory.

The structure of homoaug_dir is like this:

```
homoaug_dir
├── PDBID
│   ├── PDBID.fasta
│   ├── PDBID.pdb
│   ├── PDBID_pocket.pdb
│   ├── PDBID_pocket_chain.pdb
│   ├── PDBID_ligand.pdb
│   ├── PDBID_pocket_position.txt
│   ├── rotation_matrix
│   │   ├── (several TMalign output files)
│   ├── *extend*
│       ├──AugmentedPDBID1
│       │   ├── AugmentedPDBID1_protein.pdb
│       │   ├── AugmentedPDBID1_pocket.pdb
│       ├── AugmentedPDBID2
│       │   ├── AugmentedPDBID2_protein.pdb
│       │   ├── AugmentedPDBID2_pocket.pdb
│       ......
│     
......
```

Each AugmentedPDBID in extend directory refers to an extended pocket-ligand pair.

#### --fasta_file

The fasta file crossponding to the id_file. Each title should have the pdbid at the beginning.

e.g.:

```
>1q20_A_O00204_A_PLO 
SDISEISQKLPGEYFRYK......
>1q21_A_P01112_A_GDP 
MTEYKLVVVGAGGVGKSA......
>1q22_A_O00204_A_A3P 
SDISEISQKLPGEYFRVP......
......
```

#### --protein_pdb_dir

CIF files of the input proteins, in cif format.

Name format: <pdb_id>.cif

#### --pocket_pdbs_dir

PDB files of the input pockets, in pdb format.

Name format: <pdb_id>.pdb

#### --jackhmmer_output_dir

The temporary directory for storing the output of jackhmmer.

#### --n_thread

Number of threads for running.

#### --database_fasta_path

The fasta file of the database used for augmentation database, e.g. AF2DB.

#### --database_pdb_dir

The pdb directory of the database used for augmentation database, e.g. AF2DB's pdb directory.

#### --max_extend_num

The max number of extended pocket-ligand pairs for one real pocket-ligand pair.

#### --TMscore_threshold and Match_rate_threshold

Only the extended pocket-ligand pairs with TMscore >= TMscore_threshold and Match_rate >= Match_rate_threshold will be kept.

Match_rate is the ratio of the number of matched residues to the number of residues in the real pocket.

## Note

For various error scenarios, we choose to skip the HomoAug for this pocket-ligand pair, including but not limited to cases where the pocket is composed of multiple chains. Please note that even if the program executes successfully, some files may be missing in certain output directories, indicating that this pocket-ligand pair is currently not suitable for HomoAug. In summary, all valid HomoAug results will be saved in the 'extend' folder.