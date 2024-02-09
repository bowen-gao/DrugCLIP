# Copyright © 2023 Institute for AI Industry Research (AIR), Tsinghua University.
# License: GNU GPLv3. [See details in LICENSE]

import argparse
import os
import glob
import Bio.PDB
import ray
import random
import json
import mmap
import time
import sys
import traceback
from ray.util.queue import Queue
import numpy as np

sys.path.append(".")
from utils.misc import execute
from utils.ray_tools import ProgressBar
from tqdm import tqdm
import pathlib
import subprocess
import shutil


class JackhmmerRunner:
    def __init__(self, database_dir, task_ids, seq_dir, output_dir, n_thread):
        self.N_CPU_PER_THREAD = 1
        self.n_thread = n_thread
        self.database_dir = database_dir
        self.task_ids = task_ids
        self.seq_dir = seq_dir
        self.output_dir = output_dir
        if output_dir.endswith("/"):
            output_dir = output_dir[:-1]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.sto_dir = output_dir + "_sto"
        if not os.path.exists(self.sto_dir):
            os.mkdir(self.sto_dir)
        self.a2m_dir = output_dir + "_a2m"
        if not os.path.exists(self.a2m_dir):
            os.mkdir(self.a2m_dir)

    def split_list(_list, n):
        chunk_size = (len(_list) - 1) // n + 1
        chunks = [_list[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]
        return chunks

    @ray.remote(num_cpus=1)
    def process_jobs(self, id, jobs_queue, actor):
        print("start process", id)
        while not jobs_queue.empty():
            job = jobs_queue.get()
            try:
                self.execute_one_job(job)

            except:
                print(f"failed: {job}")
                traceback.print_exception(*sys.exc_info())
            try:
                actor.update.remote(1)
            except:
                pass
        return 1

    def execute_one_job(self, job):
        seq_file_Path = pathlib.Path(self.seq_dir) / job[:4] / (job + ".fasta")
        output_file_Path = pathlib.Path(self.sto_dir) / (job + ".sto")
        execute(
            f"jackhmmer"
            f" --cpu {self.N_CPU_PER_THREAD}"
            f" -A {output_file_Path}"
            f" -o /dev/null"
            f" -E 0.001"
            f" -N 3"
            f" {str(seq_file_Path)}"
            f" {self.database_dir}"
        )
        # with open(output_file_Path,"w") as f:
        #     f.write("")
        # # sleep 1 s
        # time.sleep(1)
        return 1

    def change_sto_to_fasta(self):
        sto_files = glob.glob(self.sto_dir + "/*.sto")
        for sto_path in tqdm(sto_files):
            pdb_id = sto_path.split("/")[-1].split(".")[0]
            # print(pdb_id)
            a2m_path = self.a2m_dir + f"/{pdb_id}.a2m"
            execute(
                f"esl-reformat --informat stockholm"
                f" -o {str(a2m_path)} a2m"
                f" {str(sto_path)}"
            )
            fasta_path = self.output_dir + f"/{pdb_id}.fasta"
            output = ""
            with open(a2m_path) as f:
                for line in f:
                    line = line.strip()
                    if line[0] != ">":
                        output += line
                    else:
                        output += "\n" + line + "\n"
            output = output.strip()
            output = output.split("\n")[:-2]
            output = "\n".join(output)
            with open(fasta_path, "w") as f:
                f.write(output)

    def run_jackhmmer(self):
        all_jobs = []
        with open(self.task_ids, "r") as f:
            data = f.readlines()
            for line in data:
                job = line.strip()
                job = job[:4].upper()
                all_jobs.append(job)
        print("all jobs:", len(all_jobs))
        uncompleted_jobs = all_jobs
        # completed?
        # uncompleted_jobs=[]
        # for job in all_jobs:
        #     if (not is_complete(job)):
        #         uncompleted_jobs.append(job)
        # print("uncompleted jobs:",len(uncompleted_jobs))
        ray.init()
        job_queue = Queue()
        for job in tqdm(uncompleted_jobs):
            job_queue.put(job)
        print("job queue size:", job_queue.qsize())
        pb = ProgressBar(len(all_jobs))
        actor = pb.actor
        print("actor:", actor)
        job_id_list = []
        self.n_thread = min(self.n_thread, len(uncompleted_jobs))
        for i in range(self.n_thread):
            job_id_list.append(self.process_jobs.remote(self, i, job_queue, actor))
        pb.print_until_done()
        result = ray.get(job_id_list)
        print("Run homo search done!")
        ray.shutdown()

        # remove tmp folders
        shutil.rmtree(self.a2m_dir)
        shutil.rmtree(self.sto_dir)

        # change sto to fasta
        self.change_sto_to_fasta()


class LigandPocketExtractor:
    def __init__(self, id_file, homoaug_dir, n_thread):
        self.id_file = id_file
        self.homoaug_dir = homoaug_dir
        self.n_thread = n_thread
        self.read_ligand_name_chain_name()

    def read_ligand_name_chain_name(self):
        # read ligand_name
        ligand_name = {}
        chain_name = {}
        with open(self.id_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                id = line[:4].upper()
                ligand_name[id] = line.split("_")[-1]
                chain_name[id] = line.split("_")[1]
        # print("ligand_name:",ligand_name)
        self.ligand_name = ligand_name
        self.chain_name = chain_name

    def execute_one_job(self, job):
        id = job
        # print ('############################')
        # print (id)
        # read original pdb
        pdb_file = self.homoaug_dir + "/" + id + "/" + id + "_protein.pdb"
        if not os.path.exists(pdb_file):
            print("no pdb file for id:", id)
            return 1
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        structure = pdb_parser.get_structure(id, pdb_file)
        model = structure[0]
        chain = model[self.chain_name[id]]
        # get pocket_chain to a new pdb
        pocket_chain = Bio.PDB.Chain.Chain(self.chain_name[id])
        for residue in chain:
            if residue.id[0] == " ":
                pocket_chain.add(residue)
        io = Bio.PDB.PDBIO()
        io.set_structure(pocket_chain)
        io.save(self.homoaug_dir + "/" + id + "/" + id + "_pocket_chain.pdb")

        # save ligand to a new pdb
        ligand_chain = Bio.PDB.Chain.Chain(self.chain_name[id])
        ligand_found = False
        for residue in chain:
            if residue.resname == self.ligand_name[id]:
                ligand_chain.add(residue)
                ligand_found = True
        if ligand_found:
            io = Bio.PDB.PDBIO()
            io.set_structure(ligand_chain)
            io.save(self.homoaug_dir + "/" + id + "/" + id + "_ligand.pdb")
        else:
            print("ligand not found:", id)
            return 1

        # remove ligand in the id.pocket.pdb
        pdb_file = self.homoaug_dir + "/" + id + "/" + id + "_pocket.pdb"
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        structure = pdb_parser.get_structure(id, pdb_file)
        model = structure[0]
        chain = model["R"]
        # get pocket to a new pdb
        pocket = Bio.PDB.Chain.Chain("A")
        for residue in chain:
            if residue.id[0] == " ":
                pocket.add(residue)
        # write pocket pdb
        io = Bio.PDB.PDBIO()
        io.set_structure(pocket)
        io.save(self.homoaug_dir + "/" + id + "/" + id + "_pocket.pdb")

    @ray.remote(num_cpus=1)
    def process_jobs(self, id, jobs_queue, actor):
        print("start process", id)
        while not jobs_queue.empty():
            job = jobs_queue.get()
            try:
                self.execute_one_job(job)
            except:
                print(f"failed: {job}")
                traceback.print_exception(*sys.exc_info())
            try:
                actor.update.remote(1)
            except:
                pass
        return 1

    def run(self):
        all_jobs = glob.glob(self.homoaug_dir + "/*")
        all_jobs = [x.split("/")[-1] for x in all_jobs]
        uncompleted_jobs = all_jobs
        job_queue = Queue()
        for job in tqdm(uncompleted_jobs):
            job_queue.put(job)
        print("job queue size:", job_queue.qsize())
        pb = ProgressBar(len(all_jobs))
        actor = pb.actor
        job_id_list = []
        self.n_thread = min(self.n_thread, len(uncompleted_jobs))
        for i in range(self.n_thread):
            job_id_list.append(self.process_jobs.remote(self, i, job_queue, actor))
        pb.print_until_done()
        result = ray.get(job_id_list)
        ray.shutdown()
        print("Done!")


class PocketPositionExtractor:
    def __init__(self, homoaug_dir, n_thread):
        self.homoaug_dir = homoaug_dir
        self.n_thread = n_thread
        self.aa_3_to_1 = {
            "CYS": "C",
            "ASP": "D",
            "SER": "S",
            "GLN": "Q",
            "LYS": "K",
            "ILE": "I",
            "PRO": "P",
            "THR": "T",
            "PHE": "F",
            "ASN": "N",
            "GLY": "G",
            "HIS": "H",
            "LEU": "L",
            "ARG": "R",
            "TRP": "W",
            "ALA": "A",
            "VAL": "V",
            "GLU": "E",
            "TYR": "Y",
            "MET": "M",
            "MSE": "M",
            "CME": "C",
            "CSO": "C",
            "UNK": "X",
        }

    def execute_one_job(self, job):
        id = job
        # print ('############################')
        # print (id)

        # read pocket_pdb
        pocket_pdb_file = self.homoaug_dir + "/" + id + "/" + id + "_pocket.pdb"
        pocket_pdb_structure = Bio.PDB.PDBParser().get_structure(id, pocket_pdb_file)
        model = pocket_pdb_structure[0]
        for chain in model:
            pocket_chain_id = chain.id
            break
        chain = model[pocket_chain_id]

        # get the pocket atom coordinates
        pocket_atom_coordinates = set()
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                pocket_atom_coordinates.add(tuple(atom.get_coord()))
        # print("pocket_atom_coordinates:",pocket_atom_coordinates)

        sequence = ""
        pocket_chain_pdb_file = (
            self.homoaug_dir + "/" + id + "/" + id + "_pocket_chain.pdb"
        )
        # get the position of the pocket residues in the pocket chain
        pocket_chain_structure = Bio.PDB.PDBParser().get_structure(
            id, pocket_chain_pdb_file
        )
        model = pocket_chain_structure[0]
        for chain in model:
            pocket_chain_id = chain.id
            break
        chain_num = len(list(model.get_chains()))
        if chain_num > 1:
            print("error: more than 1 chain in pocket_pdb_file:", id)
            return 1
        chain = model[pocket_chain_id]
        for residue in chain:
            if residue.id[0] != " ":
                continue
            in_pocket = False
            for atom in residue:
                if tuple(atom.get_coord()) in pocket_atom_coordinates:
                    in_pocket = True
                    break
            if in_pocket:
                sequence += self.aa_3_to_1[residue.resname]
            else:
                sequence += "-"

        # save
        sequence_position_file = (
            self.homoaug_dir + "/" + id + "/" + id + "_pocket_position.txt"
        )
        with open(sequence_position_file, "w") as f:
            f.write(sequence)

    @ray.remote(num_cpus=1)
    def process_jobs(self, id, jobs_queue, actor):
        print("start process", id)
        while not jobs_queue.empty():
            job = jobs_queue.get()
            try:
                self.execute_one_job(job)
            except:
                print(f"failed: {job}")
                traceback.print_exception(*sys.exc_info())
            try:
                actor.update.remote(1)
            except:
                pass
        return 1

    def run(self):
        all_jobs = []
        for id in os.listdir(self.homoaug_dir):
            if os.path.isfile(self.homoaug_dir + "/" + id + "/" + id + "_ligand.pdb"):
                if os.path.isfile(
                    self.homoaug_dir + "/" + id + "/" + id + "_pocket.pdb"
                ):
                    all_jobs.append(id)
        uncompleted_jobs = all_jobs
        job_queue = Queue()
        for job in tqdm(uncompleted_jobs):
            job_queue.put(job)
        print("job queue size:", job_queue.qsize())
        pb = ProgressBar(len(all_jobs))
        actor = pb.actor
        job_id_list = []
        self.n_thread = min(self.n_thread, len(uncompleted_jobs))
        for i in range(self.n_thread):
            job_id_list.append(self.process_jobs.remote(self, i, job_queue, actor))
        pb.print_until_done()
        result = ray.get(job_id_list)
        ray.shutdown()
        print("Done!")


class TMalignRunner:
    def __init__(
        self,
        max_extend_num,
        homoaug_dir,
        MSA_dir,
        AF2DB_dir,
        TMscore_threshold,
        Match_rate_threshold,
        n_thread,
    ):
        self.n_thread = n_thread
        self.max_extend_num = max_extend_num
        self.homoaug_dir = homoaug_dir
        self.MSA_dir = MSA_dir
        self.AF2DB_dir = AF2DB_dir
        self.TMscore_threshold = TMscore_threshold
        self.Match_rate_threshold = Match_rate_threshold

    def _remove_gap_of_primary_sequence(self, primary_sequence, candidate_sequence):
        assert len(primary_sequence) == len(candidate_sequence)
        primary_sequence_without_gap = ""
        candidate_sequence_without_gap = ""
        for i in range(len(primary_sequence)):
            if primary_sequence[i] != "-":
                primary_sequence_without_gap += primary_sequence[i]
                candidate_sequence_without_gap += candidate_sequence[i]
        return primary_sequence_without_gap, candidate_sequence_without_gap

    def _calc_match_rate(self, pocket_position, Aligned_seq):
        total_cnt = 0
        match_cnt = 0
        for i in range(len(Aligned_seq)):
            if pocket_position[i] != "-":
                total_cnt += 1
                if pocket_position[i] == Aligned_seq[i]:
                    match_cnt += 1
        return match_cnt / total_cnt

    def _get_rotate_matrix(self, rotate_matrix_file):
        with open(rotate_matrix_file, "r") as f:
            data = f.readlines()
        u = []
        t = []
        for i in range(2, 5):
            line = data[i].split(" ")
            line_float = [float(x) for x in line if x != ""]
            t.append(line_float[1])
            u.append(line_float[2:])
        u = np.array(u)
        t = np.array(t)
        return u, t

    def _read_ligand_pdb(self, ligand_pdb_file):
        parser = Bio.PDB.PDBParser()
        structure = parser.get_structure("ligand", ligand_pdb_file)
        ligand_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        ligand_coords.append(atom.get_coord())
        return ligand_coords

    @ray.remote(num_cpus=1)
    def process_jobs(self, id, jobs_queue, actor):
        print("start process", id)
        while not jobs_queue.empty():
            job = jobs_queue.get()
            try:
                self.execute_one_job(job)

            except:
                print(f"failed: {job}")
                traceback.print_exception(*sys.exc_info())
            try:
                actor.update.remote(1)
            except:
                pass
        return 1

    def execute_one_job(self, id):
        # print("#######################")
        print(id)
        # get the sequence from pdb
        fasta_dir = self.homoaug_dir + "/" + id + "/" + id + ".fasta"
        with open(fasta_dir) as f:
            fasta = f.readlines()

        # read the pocket position
        pocket_position_file = (
            self.homoaug_dir + "/" + id + "/" + id + "_pocket_position.txt"
        )
        if not os.path.exists(pocket_position_file):
            print("position_file not exist")
            return 1
        with open(pocket_position_file) as f:
            pocket_position = f.readline().strip()

        # create rotation matrix dir
        rotation_matrix_dir = self.homoaug_dir + "/" + id + "/" + "rotation_matrix/"
        if not os.path.exists(rotation_matrix_dir):
            os.makedirs(rotation_matrix_dir)

        # get the sequence from TMalign
        chain_pdb_file = self.homoaug_dir + "/" + id + "/" + id + "_pocket_chain.pdb"
        MSA_file = self.MSA_dir + f"/{id}" + ".fasta"
        if not os.path.exists(MSA_file):
            print("MSA_file not exist")
            return 1
        MSA_ids = []
        with open(MSA_file) as f:
            lines = f.readlines()
            for idx in range(0, len(lines), 2):
                MSA_ids.append(lines[idx].strip().split(" ")[-1])

        # create extend dir
        extend_dir = self.homoaug_dir + "/" + id + "/" + "extend"
        if not os.path.exists(extend_dir):
            os.mkdir(extend_dir)

        # get ligand
        ligand_file = self.homoaug_dir + "/" + id + "/" + id + "_ligand.pdb"
        ligand_coords = self._read_ligand_pdb(ligand_file)

        # TMalign
        cnt = len(glob.glob(extend_dir + "/*"))
        for MSA_id in list(MSA_ids):
            if cnt >= self.max_extend_num:
                break
            # calculate TMscore
            MSA_pdb_file = self.AF2DB_dir + f"/{MSA_id}.pdb"
            if not os.path.exists(MSA_pdb_file):
                continue
            rotation_matrix_file = rotation_matrix_dir + f"{MSA_id}.txt"
            if os.path.exists(rotation_matrix_file):
                continue
            out_bytes = subprocess.check_output(
                ["TMalign", MSA_pdb_file, chain_pdb_file, "-m", rotation_matrix_file]
            )
            out_text = out_bytes.decode("utf-8").strip().split("\n")
            TMscore1 = float(out_text[12].split(" ")[1])
            TMscore2 = float(out_text[13].split(" ")[1])
            (
                sequence_from_TMalign,
                MSA_aligned_sequence,
            ) = self._remove_gap_of_primary_sequence(out_text[19], out_text[17])
            TMalign_file = rotation_matrix_dir + f"{MSA_id}_TMscore.txt"
            with open(TMalign_file, "w") as f:
                f.write("TMscore normalized to chain_pdb:" + str(TMscore2) + "\n")
                f.write("TMscore normalized to MSA_pdb:" + str(TMscore1) + "\n")
                f.write("Aligned sequence : \n")
                f.write(sequence_from_TMalign + "\n")
                f.write(MSA_aligned_sequence + "\n")
            TMscore = TMscore2

            # calculate Match_score
            Aligned_seq = MSA_aligned_sequence
            Match_rate = self._calc_match_rate(pocket_position, Aligned_seq)

            # print("MSA_id:",MSA_id)
            # print("TMscore:",TMscore)
            # print("Match_rate:",Match_rate)
            if (
                TMscore >= self.TMscore_threshold
                and Match_rate >= self.Match_rate_threshold
            ):
                extend_instance_dir = extend_dir + "/" + MSA_id + "/"
                # if os.path.exists(extend_instance_dir):
                #     continue
                os.mkdir(extend_instance_dir)

                # read ori MSA pdb file
                MSA_pdb_file = self.AF2DB_dir + f"/{MSA_id}" + ".pdb"
                parser = Bio.PDB.PDBParser()
                structure = parser.get_structure(MSA_id, MSA_pdb_file)
                model = structure[0]
                for chain in model:
                    MSA_chain_id = chain.id
                    break
                MSA_chain = model[MSA_chain_id]

                # get rotate_matrix
                rotation_matrix_file = rotation_matrix_dir + f"{MSA_id}.txt"
                rotation_matrix = self._get_rotate_matrix(rotation_matrix_file)

                for residue in MSA_chain:
                    for atom in residue:
                        coord = atom.get_coord()
                        coord = np.array(coord)
                        new_coord = (
                            np.dot(rotation_matrix[0], coord) + rotation_matrix[1]
                        )
                        atom.set_coord(new_coord)

                # write new pdb file
                io = Bio.PDB.PDBIO()
                io.set_structure(structure)
                io.save(extend_instance_dir + f"{MSA_id}" + "_protein.pdb")

                # get pocket , which is in the 6A of ligand
                MSA_pocket_file = extend_instance_dir + f"{MSA_id}" + "_pocket.pdb"
                for residue in MSA_chain:
                    remove_atom_ids = []
                    for atom in residue:
                        # print("atom: ",atom.id)
                        coord = atom.get_coord()
                        f = 0
                        for ligand_coord in ligand_coords:
                            dis = np.linalg.norm(coord - ligand_coord)
                            if np.linalg.norm(coord - ligand_coord) <= 6:
                                f = 1
                                break
                        if f == 0:
                            remove_atom_ids.append(atom.id)
                    for atom_id in remove_atom_ids:
                        residue.detach_child(atom_id)
                io = Bio.PDB.PDBIO()
                io.set_structure(structure)
                io.save(MSA_pocket_file)
                cnt += 1
        print("finish: pdb_id:", id)
        return 1

    def run(self):
        all_jobs = []
        for id in os.listdir(self.homoaug_dir):
            if (
                os.path.isfile(self.homoaug_dir + "/" + id + "/" + id + "_ligand.pdb")
                and os.path.isfile(
                    self.homoaug_dir + "/" + id + "/" + id + "_pocket.pdb"
                )
                and os.path.isfile(
                    self.homoaug_dir + "/" + id + "/" + id + "_pocket_position.txt"
                )
            ):
                all_jobs.append(id)
        uncompleted_jobs = all_jobs
        job_queue = Queue()
        for job in tqdm(uncompleted_jobs):
            job_queue.put(job)
        print("job queue size:", job_queue.qsize())
        pb = ProgressBar(len(all_jobs))
        actor = pb.actor
        job_id_list = []
        self.n_thread = min(self.n_thread, len(uncompleted_jobs))
        for i in range(self.n_thread):
            job_id_list.append(self.process_jobs.remote(self, i, job_queue, actor))
        pb.print_until_done()
        result = ray.get(job_id_list)
        ray.shutdown()
        print("Done!")


class HomoAugRunner:
    def __init__(self, args):
        self.id_file = args.id_file
        self.homoaug_dir = args.homoaug_dir
        self.fasta_file = args.fasta_file
        self.protein_pdb_dir = args.protein_pdb_dir
        self.pocket_pdbs_dir = args.pocket_pdbs_dir
        self.jackhmmer_output_dir = args.jackhmmer_output_dir
        self.database_fasta_path = args.database_fasta_path
        self.max_extend_num = args.max_extend_num
        self.database_pdb_dir = args.database_pdb_dir
        self.TMscore_threshold = args.TMscore_threshold
        self.Match_rate_threshold = args.Match_rate_threshold
        self.n_thread = args.n_thread

    def read_dataset_fasta(self):
        protein_seq = {}
        with open(self.fasta_file) as f:
            fasta = f.readlines()
            for i in range(0, len(fasta), 2):
                id = fasta[i].strip()
                id = id[1:5].upper()
                seq = fasta[i + 1].strip()
                protein_seq[id] = seq
        return protein_seq

    def create_dir(self):
        # Create homoaug dir and subdirs
        # Dir format
        # homoaug_dir
        # └── id
        #     ├── id.fasta
        #     └── id_protein.pdb
        #     └── id_pocket.pdb

        protein_seq = self.read_dataset_fasta()
        if not os.path.exists(self.homoaug_dir):
            os.mkdir(self.homoaug_dir)
        with open(self.id_file, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):
            id = line.strip()
            id = id[:4].upper()
            if not os.path.exists(self.homoaug_dir + "/" + id):
                os.mkdir(self.homoaug_dir + "/" + id)

                # create fasta
                with open(self.homoaug_dir + "/" + id + "/" + id + ".fasta", "w") as f:
                    f.write(">" + id + "\n" + protein_seq[id] + "\n")

                # copy pdb
                cif_file = glob.glob(self.protein_pdb_dir + "/" + id + "*")
                if len(cif_file) != 1:
                    print("error : having more than 1 cif file", id)
                    continue
                cif_file = cif_file[0]
                # read cif
                parser = Bio.PDB.MMCIFParser(QUIET=True)
                structure = parser.get_structure(id, cif_file)
                # get number of models
                n_model = len(structure)
                if n_model != 1:
                    print("error : having more than 1 model", id)
                    continue
                # get number of chains
                n_chain = len(list(structure.get_chains()))
                if n_chain != 1:
                    print("error : having more than 1 chain", id)
                    continue
                # save pdb
                io = Bio.PDB.PDBIO()
                io.set_structure(structure)
                io.save(self.homoaug_dir + "/" + id + "/" + id + "_protein.pdb")
                # os.system("cp "+homoaug_dir+"/"+id+"/"+id+"_protein.pdb "+homoaug_dir+"/"+id+"/"+id+"_pocket_chain.pdb")

                # copy pocket
                pocket_file = glob.glob(self.pocket_pdbs_dir + "/" + id + "*")
                if len(pocket_file) != 1:
                    print("error : having more than 1 pocket file", id)
                    continue
                pocket_file = pocket_file[0]
                # copy to homoaug
                os.system(
                    "cp "
                    + pocket_file
                    + " "
                    + self.homoaug_dir
                    + "/"
                    + id
                    + "/"
                    + id
                    + "_pocket.pdb"
                )

    def run(self):
        self.create_dir()

        # Run jackhmmer
        print("# Start running jackhmmer")
        jackhmmer_runner = JackhmmerRunner(
            database_dir=self.database_fasta_path,
            task_ids=self.id_file,
            seq_dir=self.homoaug_dir,
            output_dir=self.jackhmmer_output_dir,
            n_thread=self.n_thread,
        )
        jackhmmer_runner.run_jackhmmer()

        print("# Start running ligand pocket extractor")
        ligand_pocket_extractor = LigandPocketExtractor(
            self.id_file, self.homoaug_dir, self.n_thread
        )
        ligand_pocket_extractor.run()

        print("# Start running pocket position extractor")
        pocket_position_extractor = PocketPositionExtractor(
            self.homoaug_dir, self.n_thread
        )
        pocket_position_extractor.run()

        print("# Start running TMalign")
        tmalign_runner = TMalignRunner(
            self.max_extend_num,
            self.homoaug_dir,
            self.jackhmmer_output_dir,
            self.database_pdb_dir,
            self.TMscore_threshold,
            self.Match_rate_threshold,
            self.n_thread,
        )
        tmalign_runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()
    homoaug_runner = HomoAugRunner(args)
    homoaug_runner.run()
    print("HomoAug Done!")