# http://www.sgdml.org/#code
# https://github.com/dkhan42/MBDF/tree/main
import random
import numpy as np
random.seed(666)
np.random.seed(666)
import os
import pdb

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import deepchem.molnet as mn
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles
import re
import bz2, pickle
import json
import deepchem as dc

from vec2str import ZipFeaturizer
from representations import get_cMBDF, get_all_slatm, gen_all_bob


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

        if not os.path.exists(f'./.data/{self.molname}.npz'):
            print('Downloading the file')
            import urllib.request
            url = f'http://www.quantum-machine.org/gdml/data/npz/md17_{self.molname}.npz'
            os.makedirs('.data', exist_ok=True)
            urllib.request.urlretrieve(url, f'.data/{self.molname}.npz')
            print('Download complete')
        else:
            print('File already exists')

        data = np.load(f'./.data/{self.molname}.npz')

        indices = np.random.choice(data["R"].shape[0], 20000, replace=False)
        self.z = (np.array(list(data["z"]) * data["R"].shape[0])).reshape(
            data["R"].shape[0], -1
        )[indices]

        self.R = data['R'][indices]
        self.E = data['E'][indices]

    def gen_representation(self):
        """
        generate the representation of the molecule
        """
        X_MBDF = get_cMBDF(self.z, self.R, local=False)
        X_SLATM = get_all_slatm(self.z, self.R, local=False)
        X_BOB = gen_all_bob(self.R, self.z, size=100, asize={"O": 4, "C": 12, "N": 3, "H": 16, "S": 1})
        
        self.results = {"cMBDF": X_MBDF, "SLATM": X_SLATM, "BOB": X_BOB, "y": self.E}

        return self.results

if __name__ == '__main__':

    featurizer_mapping = {
    'rdkit': dc.feat.RDKitDescriptors(),
    'mol2vec': dc.feat.Mol2VecFingerprint(),
    'ecfp': dc.feat.CircularFingerprint(size=2048, radius=4),
    'mordred': dc.feat.MordredDescriptors(ignore_3D=True)
    }

    smallMol = SmallMolTraj("aspirin")
    smallMol.get_data()
    smallMol.gen_representation()

    tasks, X_train, y_train, X_valid, y_valid, X_test, y_test = molnet_loader(
        "qm7", preproc=False
    )
    print(X_train[1], y_train[1])
    print(X_valid[1], y_valid[1])
    print(X_test[1], y_test[1])
    print(tasks)
    X_feat_train, y_train = dc_featurize(X_train, y_train, featurizer_mapping['rdkit'])
    X_feat_valid, y_valid = dc_featurize(X_valid, y_valid, featurizer_mapping['rdkit'])
    X_feat_test, y_test = dc_featurize(X_test, y_test, featurizer_mapping['rdkit'])
    print(X_feat_test.shape)
    #print(X_feat_test[1], y_test[1])   
    converter = ZipFeaturizer()
    string_reps = converter.bin_vectors(X_feat_test)
    print(string_reps[0])
    pdb.set_trace()
