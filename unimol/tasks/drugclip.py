# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from IPython import embed as debug_embedded
import logging
import os
from collections.abc import Iterable
from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset,LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset, AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
logger = logging.getLogger(__name__)


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio*n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp>= num:
                break
    return (tp*n)/(p*fp)


def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    #print(fpr, tpr)
    res = {}
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)

    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             #print(fpr[i], tpr[i])
    #             if fpr[i-1]==0:
    #                 res[str(ratio)]=tpr[i]/fpr[i]
    #             else:
    #                 res[str(ratio)]=tpr[i-1]/fpr[i-1]
    #             break
    
    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    #print(res)
    #print(res2)
    return res2

def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    
        # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index  = np.argsort(y_score)[::-1]
    for i in range(int(len(index)*0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list



@register_task("drugclip")
class DrugCLIP(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.mol_reps = None
        self.keys = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")
            
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            
        else:
            
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")


        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),
                    "pocket_len": RawArrayDataset(pocket_len_dataset)
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset


    

    def load_mols_dataset(self, data_path,atoms,coords, **kwargs):
 
        dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "target":  RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset
    

    def load_retrieval_mols_dataset(self, data_path,atoms,coords, **kwargs):
 
        dataset = LMDBDataset(data_path)
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def load_pockets_dataset(self, data_path, **kwargs):

        dataset = LMDBDataset(data_path)
 
        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )




        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")



        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
            
        if args.finetune_pocket_model is not None:
            print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)

        return model

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output


    def test_pcba_target(self, name, model, **kwargs):
        """Encode a dataset with the molecule encoder."""

        #names = "PPARG"
        data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=64
        #print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)

            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = "/data/protein/lib-pcba/raw/lib_pcba/" + name + "/pockets.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_names = sample["pocket_name"]
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)

        res = pocket_reps @ mol_reps.T
        res_single = res.max(axis=0)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        return auc, bedroc, ef_list, re_list
    
    
    

    def test_pcba(self, model, **kwargs):
        #ckpt_date = self.args.finetune_from_model.split("/")[-2]
        #save_name = "/home/gaobowen/DrugClip/test_results/pcba/" + ckpt_date + ".txt"
        save_name = ""
        
        targets = os.listdir("/data/protein/lib-pcba/raw/lib_pcba/")

        print(targets)
        auc_list = []
        ef_list = []
        bedroc_list = []

        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        for target in targets:
            auc, bedroc, ef, re = self.test_pcba_target(target, model)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            print("re", re)
            print("ef", ef)
            for key in re:
                re_list[key].append(re[key])
        print(auc_list)
        print(ef_list)
        print("auc 25%", np.percentile(auc_list, 25))
        print("auc 50%", np.percentile(auc_list, 50))
        print("auc 75%", np.percentile(auc_list, 75))
        print("auc mean", np.mean(auc_list))
        print("bedroc 25%", np.percentile(bedroc_list, 25))
        print("bedroc 50%", np.percentile(bedroc_list, 50))
        print("bedroc 75%", np.percentile(bedroc_list, 75))
        print("bedroc mean", np.mean(bedroc_list))
        #print(np.median(auc_list))
        #print(np.median(ef_list))
        for key in ef_list:
            print("ef", key, "25%", np.percentile(ef_list[key], 25))
            print("ef",key, "50%", np.percentile(ef_list[key], 50))
            print("ef",key, "75%", np.percentile(ef_list[key], 75))
            print("ef",key, "mean", np.mean(ef_list[key]))
        for key in re_list:
            print("re",key, "25%", np.percentile(re_list[key], 25))
            print("re",key, "50%", np.percentile(re_list[key], 50))
            print("re",key, "75%", np.percentile(re_list[key], 75))
            print("re",key, "mean", np.mean(re_list[key]))

        return 
    
    def test_dude_target(self, target, model, **kwargs):

        data_path = "/data/protein/DUD-E/raw/all/" + target + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=64
        print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = mol_encoder_rep
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            #print(mol_emb.dtype)
            mol_emb = mol_emb.detach().cpu().numpy()
            #print(mol_emb.dtype)
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = "/data/protein/DUD-E/raw/all/" + target + "/fpocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            #pocket_emb = pocket_encoder_rep
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        res = pocket_reps @ mol_reps.T

        res_single = res.max(axis=0)

        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)
        
        
        print(target)

        print(np.sum(labels), len(labels)-np.sum(labels))

        return auc, bedroc, ef_list, re_list, res_single, labels

    def test_dude(self, model, **kwargs):


        targets = os.listdir("/data/protein/DUD-E/raw/all/")
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list= []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        for i,target in enumerate(targets):
            auc, bedroc, ef, re, res_single, labels = self.test_dude_target(target, model)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])
            res_list.append(res_single)
            labels_list.append(labels)
        res = np.concatenate(res_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        print("auc mean", np.mean(auc_list))
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "mean", np.mean(ef_list[key]))

        for key in re_list:
            print("re", key, "mean",  np.mean(re_list[key]))

        # save printed results 
        
        
        return
    
    
    
    
    
    def encode_mols_once(self, model, data_path, emb_dir, atoms, coords, **kwargs):
        
        # cache path is embdir/data_path.pkl

        cache_path = os.path.join(emb_dir, data_path.split("/")[-1] + ".pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                mol_reps, mol_names = pickle.load(f)
            return mol_reps, mol_names

        mol_dataset = self.load_retrieval_mols_dataset(data_path,atoms,coords)
        mol_reps = []
        mol_names = []
        bsz=32
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])

        mol_reps = np.concatenate(mol_reps, axis=0)

        # save the results
        
        with open(cache_path, "wb") as f:
            pickle.dump([mol_reps, mol_names], f)

        return mol_reps, mol_names
    
    def retrieve_mols(self, model, mol_path, pocket_path, emb_dir, k, **kwargs):
 
        os.makedirs(emb_dir, exist_ok=True)        
        mol_reps, mol_names = self.encode_mols_once(model, mol_path, emb_dir,  "atoms", "coordinates")
        
        pocket_dataset = self.load_pockets_dataset(pocket_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        
        res = pocket_reps @ mol_reps.T
        res = res.max(axis=0)


        # get top k results

        
        top_k = np.argsort(res)[::-1][:k]

        # return names and scores
        
        return [mol_names[i] for i in top_k], res[top_k]


        

        
         


    

    

        
            
         

        
    
    
