# http://www.sgdml.org/#code
# https://github.com/dkhan42/MBDF/tree/main
import random
import numpy as np
random.seed(666)
np.random.seed(666)
import os

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import deepchem.molnet as mn
import pandas as pd
from pathlib import Path
from rdkit.Chem import MolFromSmiles, MolToSmiles
import re
import bz2, pickle
import json
import deepchem as dc
from tqdm import tqdm
import argparse
from selfies import encoder
from sklearn.decomposition import PCA

from Feat2LLM.vec2str import ZipFeaturizer
from Feat2LLM.representations import get_cMBDF

def numpy_encoder(obj):
    """Special JSON encoder for numpy types"""
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return json.JSONEncoder().default(obj)


def save_dict_to_json(d, filename):
    with open(filename, "w") as f:
        json.dump(d, f, default=numpy_encoder, indent=4)


def load_json_to_dict(filename, convert_to_numpy=False):
    with open(filename, "r") as f:
        d = json.load(f)

    if convert_to_numpy:

        def convert(item):
            if isinstance(item, list):
                return np.array(item)
            if isinstance(item, dict):
                return {key: convert(value) for key, value in item.items()}
            return item

        d = convert(d)

    return d


compress_fileopener = {True: bz2.BZ2File, False: open}
pkl_compress_ending = {True: ".pkl.bz2", False: ".pkl"}


def dump2pkl(obj, filename: str, compress: bool = False):
    """
    Dump an object to a pickle file.
    obj : object to be saved
    filename : name of the output file
    compress : whether bz2 library is used for compressing the file.
    """
    output_file = compress_fileopener[compress](filename, "wb")
    pickle.dump(obj, output_file)
    output_file.close()


def loadpkl(filename: str, compress: bool = False):
    """
    Load an object from a pickle file.
    filename : name of the imported file
    compress : whether bz2 compression was used in creating the loaded file.
    """
    input_file = compress_fileopener[compress](filename, "rb")
    obj = pickle.load(input_file)
    input_file.close()
    return obj


def combine_str_vec(X1, X2):
    return [x1 + x2 for x1, x2 in zip(X1, X2)]


# Adapted from https://github.com/rxn4chemistry/rxnfp
REGEX = re.compile(
    r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)


def tokenize(smiles: str) -> List[str]:
    return [token for token in REGEX.findall(smiles)]


def preprocess(smiles: str, preproc: bool = False) -> str:
    if not preproc:
        return smiles

    smiles = MolToSmiles(
        MolFromSmiles(smiles),
        kekuleSmiles=True,
        allBondsExplicit=True,
        allHsExplicit=True,
    )

    return " ".join(tokenize(smiles))


def molnet_loader(
    name: str, preproc: bool = False, task_name: str = None, **kwargs
) -> Tuple[str, np.array, np.array, np.array]:

    # https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html
    mn_loader = getattr(mn, f"load_{name}")
    dc_set = mn_loader(**kwargs)

    tasks, dataset, _ = dc_set
    train, valid, test = dataset

    y_train = train.y
    y_valid = valid.y
    y_test = test.y

    task_idx = tasks.index(task_name) if task_name in tasks else -1
    task_idx = 0

    if task_idx >= 0:
        y_train = np.expand_dims(y_train[:, task_idx], axis=1)
        y_valid = np.expand_dims(y_valid[:, task_idx], axis=1)
        y_test = np.expand_dims(y_test[:, task_idx], axis=1)

    X_train = np.array([preprocess(x, preproc) for x in train.ids])
    X_valid = np.array([preprocess(x, preproc) for x in valid.ids])
    X_test = np.array([preprocess(x, preproc) for x in test.ids])

    if name in ["freesolv", "delaney", "lipo", "bace_regression", "sampl", "qm7"]:
        # for regression tasks
        y_train = np.array(y_train, dtype=float)
        y_valid = np.array(y_valid, dtype=float)
        y_test = np.array(y_test, dtype=float)
    else:
        # for classification tasks
        y_train = np.array(y_train, dtype=int)
        y_valid = np.array(y_valid, dtype=int)
        y_test = np.array(y_test, dtype=int)

    return tasks, X_train, y_train, X_valid, y_valid, X_test, y_test

def dc_featurize(X_data, y_data, featurizer):
    features = []
    y_cleaned = []
    for index, smiles in enumerate(X_data):
        try:
            feature = featurizer.featurize(smiles)
            features.append(feature)
            y_cleaned.append(y_data[index])
        except Exception as e:
            print(f"Error featurizing SMILES: {smiles}", flush=True)
            print(f"Error message: {str(e)}", flush=True)

    features_array = np.squeeze(np.array(features))
    y_array = np.array(y_cleaned)    
    
    return features_array, y_array

def smiles_to_selfies(smiles_array):
    selfies_array = []
    for smiles_string in smiles_array:
        try:
            selfies_string = encoder(smiles_string)
            selfies_array.append(selfies_string)
        except Exception as e:
            print(f"Error converting SMILES '{smiles_string}': {e}")

    return np.array(selfies_array)

class SmallMolTraj:

    """
    intended for small molecules, learn energy from xyz coordinates of
    same molecule in different conformations
    """

    def __init__(self, molname="aspirin"):
        """
        available molecules:
        aspirin
        benzene2017
        ethanol
        malonaldehyde
        naphthalene
        salicylic
        toluene
        uracil
        """

        self.molname = molname

    def get_data(self):

        if not os.path.exists(f'./data/{self.molname}.npz'):
            print('Downloading the file')
            import urllib.request
            url = f'http://www.quantum-machine.org/gdml/data/npz/md17_{self.molname}.npz'
            os.makedirs('data', exist_ok=True)
            urllib.request.urlretrieve(url, f'data/{self.molname}.npz')
            print('Download complete')
        else:
            print('File already exists')

        data = np.load(f'./data/{self.molname}.npz')

        indices = np.random.choice(data["R"].shape[0], 80000, replace=False)
        self.z = (np.array(list(data["z"]) * data["R"].shape[0])).reshape(
            data["R"].shape[0], -1
        )[indices]

        self.R = data['R'][indices]
        self.E = data['E'][indices]

    def gen_representation(self, n_components=10):
        """
        generate the representation of the molecule
        """

        X_cMBDF         = get_cMBDF(self.z, self.R, local=False)

        # import PCA to reduce the dimensionality of the features
        pca = PCA(n_components=n_components)
        X_cMBDF_trans = pca.fit_transform(X_cMBDF)

        self.results = {
            "cMBDF": X_cMBDF,
            "cMBDF_trans": X_cMBDF_trans,
            "y": self.E,
        }

        return self.results

    def save(self):
        """
        save the representation to a file
        """
        dump2pkl(self.results, f"./data/rep_{self.molname}.pkl", compress=True)


if __name__ == '__main__':
    """
    Usage:
    python load_data.py [--mn_dataset <dataset name>] [--featurizer <featurizer>] [--optimize]

    Arguments:

    Options:
    --mn_dataset <dataset name>   Name of the MoleculeNet dataset to load
                                  Choices: qm7, delaney, lipo, tox21
                                  [default: qm7]
    --featurizer <featurizer>   Molecular featurizer to use
                                Choices: mol2vec, rdkit, ecfp, mordred
                                [default: rdkit]
    --do_small                  Load MD trajectory datasets

    Example:
    python load_data.py --mn_dataset qm7 --featurizer mol2vec --optimize
    """
    parser = argparse.ArgumentParser(description='Generate dataset features', usage=__doc__)
    parser.add_argument('--mn_dataset', type=str, default='qm7',choices=['qm7', 'delaney', 'lipo', 'tox21'] ,help='Name of the MoleculeNet dataset to load')
    parser.add_argument('--featurizer', type=str, default='rdkit', choices=['mol2vec', 'rdkit', 'ecfp', 'mordred'], help='Molecular featurizer to use (default: rdkit)')
    parser.add_argument('--do_small', action='store_true', help='Load MD trajectory dataset features')
    parser.add_argument('--do_smiles', action='store_true', help='Save SMILES data')
    parser.add_argument("--n_components",type=int,default=10,help="n_components for PCA in the cMBDF representation", required=False)

    args = parser.parse_args()

    if args.do_small:
        SMALL_MOLECULES = ["aspirin", "benzene2017", "ethanol"]

        for mol in tqdm(SMALL_MOLECULES):
            smallMol = SmallMolTraj(mol)
            smallMol.get_data()
            smallMol.gen_representation(args.n_components)
            smallMol.save()

    elif args.do_smiles:
        tasks, X_train, y_train, X_valid, y_valid, X_test, y_test = molnet_loader(
            args.mn_dataset, preproc=False
        )
        X_train_selfies = smiles_to_selfies(X_train)
        X_valid_selfies = smiles_to_selfies(X_valid)
        X_test_selfies  = smiles_to_selfies(X_test)
        results = {"X_train": X_train, "X_valid": X_valid, "X_test": X_test, "y_train": y_train, "y_valid": y_valid, "y_test": y_test, "X_train_selfies": X_train_selfies, "X_valid_selfies": X_valid_selfies, "X_test_selfies": X_test_selfies}
        dump2pkl(results, f"./data/rep_{args.mn_dataset}_smiles_selfies.pkl", compress=True)

    else:
        featurizer_mapping = {
        'rdkit': dc.feat.RDKitDescriptors(),
        'mol2vec': dc.feat.Mol2VecFingerprint(),
        'ecfp': dc.feat.CircularFingerprint(size=2048, radius=4),
        'mordred': dc.feat.MordredDescriptors(ignore_3D=True)
        }
        tasks, X_train, y_train, X_valid, y_valid, X_test, y_test = molnet_loader(
            args.mn_dataset, preproc=True
        )
        print(X_train[1], y_train[1])
        print(X_valid[1], y_valid[1])
        print(X_test[1], y_test[1])

        X_feat_train, y_train = dc_featurize(X_train, y_train, featurizer_mapping[args.featurizer])
        X_feat_valid, y_valid = dc_featurize(X_valid, y_valid, featurizer_mapping[args.featurizer])
        X_feat_test, y_test   = dc_featurize(X_test, y_test, featurizer_mapping[args.featurizer])
        print(X_feat_test.shape)

        converter = ZipFeaturizer(n_bins=300)

        X_feat_train_str = converter.bin_vectors(X_feat_train)
        X_feat_valid_str = converter.bin_vectors(X_feat_valid)
        X_feat_test_str = converter.bin_vectors(X_feat_test)

        ## TODO add a PCA here

        X_train_combine =   combine_str_vec(X_train, X_feat_train_str)
        X_valid_combine =   combine_str_vec(X_valid, X_feat_valid_str)
        X_test_combine  =   combine_str_vec(X_test, X_feat_test_str)

        results = {"X_train": X_feat_train, "X_valid": X_feat_valid, "X_test": X_feat_test,
                   "X_train_str": X_feat_train_str, "X_valid_str": X_feat_valid_str, "X_test_str": X_feat_test_str,
                   "X_train_combine": X_train_combine, "X_valid_combine": X_valid_combine, "X_test_combine": X_test_combine,
                   "y_train": y_train, "y_valid": y_valid, "y_test": y_test}

        dump2pkl(results, f"./data/rep_{args.mn_dataset}_{args.featurizer}.pkl", compress=True)
